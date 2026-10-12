"""Fed-ISIC2019 data: FLamby's split and transforms, plus the 18-client partition.

FLamby (flamby/datasets/fed_isic2019/dataset.py) reads the preprocessed images
(shorter side 224 px, colour constancy -- see ../cfl_isic/download_preprocess.py) and
applies, per sample, with albumentations:

    train: RandomScale(0.07), Rotate(50), RandomBrightnessContrast(0.15, 0.1),
           Flip(p=.5), Affine(shear=0.1), RandomCrop(200, 200),
           CoarseDropout(randint(1, 8), 16, 16), Normalize()   [ImageNet mean/std]
    test : CenterCrop(200, 200), Normalize()

image -> np.transpose(2, 0, 1) -> float32 tensor; targets are ints 0..7.
Centre and train/test come from ``dataset_creation_scripts/train_test_split``
(columns image, target, center, fold, fold2): 6 centres, 18 413 train / 4 650 test.

Two implementation choices here, both documented in README_ISIC-style detail:
  * every image is centre-cropped ONCE to 220x220 and cached in a uint8 memmap
    (3.2 GB, ``isic/cache_220.npy``).  Test crops (centre 200) are pixel-identical
    to FLamby; train crops are drawn from the central 220x220 rather than the full
    ~224x298 image;
  * the train augmentations are re-implemented as ONE batched random affine
    (scale x rotation x shear x translation-crop, reflection padding = the
    albumentations default) plus flips, brightness/contrast and coarse dropout, run
    on the GPU.  Doing it per-sample on the CPU would dominate the runtime.

THE 18-CLIENT PARTITION
-----------------------
Each centre's TRAIN set is split into ``clients_per_center`` sub-clients with the
recipe of ../cfl-replicate-CIFAR (cfl_cifar/data_utils.py::partition):

    idcs = rng.permutation(n)           # one seeded shuffle
    blocks = contiguous slices of idcs  # disjoint, (near-)equal, unstratified

applied ONCE PER CENTRE, so a sub-client only ever holds images of its own hospital.
Client i belongs to centre ``i // clients_per_center``, and that centre is the
GROUND-TRUTH CLUSTER used to score the recovered partition (ARI, purity, gap_true).

  BCN        9930 -> 3310 / 3310 / 3310   clients  0, 1, 2
  molemax    3163 -> 1055 / 1054 / 1054   clients  3, 4, 5
  modern     2691 ->  897 /  897 /  897   clients  6, 7, 8
  rosendahl  1807 ->  603 /  602 /  602   clients  9,10,11
  MSK         655 ->  219 /  218 /  218   clients 12,13,14
  vienna      351 ->  117 /  117 /  117   clients 15,16,17
"""
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

CENTER_NAMES = ["BCN", "HAM_vidir_molemax", "HAM_vidir_modern", "HAM_rosendahl",
                "MSK", "HAM_vienna_dias"]
SHORT_NAMES = ["BCN", "molemax", "modern", "rosendahl", "MSK", "vienna"]
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


# ------------------------------------------------------------------- the cache
def build_cache(root, size=220, log=print):
    """-> (metadata DataFrame, uint8 memmap (N, size, size, 3)).

    Built once from the preprocessed jpgs; ~3.2 GB for 23 247 images.
    """
    meta = pd.read_csv(os.path.join(root, "train_test_split.csv"))
    path = os.path.join(root, "cache_{}.npy".format(size))
    done = path + ".done"
    if os.path.exists(done):
        return meta, np.load(path, mmap_mode="r")
    src = os.path.join(root, "ISIC_2019_Training_Input_preprocessed")
    log("building image cache {} ({} images, {:.1f} GB)".format(
        path, len(meta), len(meta) * size * size * 3 / 1e9))
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                    shape=(len(meta), size, size, 3))
    for i, name in enumerate(meta["image"]):
        img = np.asarray(Image.open(os.path.join(src, name + ".jpg")).convert("RGB"))
        h, w = img.shape[:2]
        if min(h, w) < size:
            raise ValueError("{} is {}x{} < {}".format(name, h, w, size))
        t, l = (h - size) // 2, (w - size) // 2
        arr[i] = img[t:t + size, l:l + size]
        if (i + 1) % 2000 == 0:
            log("  cached {}/{}".format(i + 1, len(meta)))
    arr.flush()
    del arr
    open(done, "w").close()
    return meta, np.load(path, mmap_mode="r")


# --------------------------------------------------------- the CIFAR-style split
def partition_pool(idx, n_parts, rng, mode="array_split"):
    """One seeded shuffle, then contiguous blocks (cfl_cifar/data_utils.py::partition).

    array_split : near-equal blocks, every image used
    equal_drop  : CIFAR arithmetic ``per = n // n_parts``; the remainder is dropped
    """
    idx = np.asarray(idx)
    shuffled = idx[rng.permutation(len(idx))]
    if mode == "equal_drop":
        per = len(idx) // n_parts
        return [np.sort(shuffled[i * per:(i + 1) * per]) for i in range(n_parts)]
    if mode != "array_split":
        raise ValueError("partition_mode must be 'array_split' or 'equal_drop'")
    return [np.sort(p) for p in np.array_split(shuffled, n_parts)]


# ------------------------------------------------------------ client datasets
class ClientData:
    """Indices into the shared memmap, with FLamby's DataLoaderWithMemory semantics."""

    def __init__(self, images, idx, labels, batch_size, seed, name, train):
        self.images, self.idx, self.name, self.train = images, np.asarray(idx), name, train
        self.indices = self.idx.tolist()          # kept so a run can be replayed exactly
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(int(seed))
        self._order, self._pos = None, 0

    def __len__(self):
        return len(self.idx)

    def _load(self, j):
        j = np.asarray(j)
        srt = np.argsort(self.idx[j])             # sequential memmap reads are faster
        x = torch.from_numpy(np.ascontiguousarray(self.images[self.idx[j][srt]]))
        inv = torch.from_numpy(np.argsort(srt))
        return x[inv].permute(0, 3, 1, 2), self.labels[torch.from_numpy(j)]

    def next_batch(self):
        if self._order is None or self._pos >= len(self._order):
            self._order, self._pos = self.rng.permutation(len(self.idx)), 0
        j = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return self._load(j)

    def batches(self, batch_size=None):
        bs = batch_size or self.batch_size
        for s in range(0, len(self.idx), bs):
            yield self._load(np.arange(s, min(len(self.idx), s + bs)))


# ----------------------------------------------------------------- transforms
def normalize(x):
    return (x.float() - MEAN.to(x.device)) / STD.to(x.device)


def test_transform(x_uint8, crop=200):
    s = x_uint8.shape[-1]
    o = (s - crop) // 2
    return normalize(x_uint8[..., o:o + crop, o:o + crop])


def train_transform(x_uint8, gen, crop=200):
    """Batched GPU version of FLamby's albumentations train pipeline."""
    n, _, s, _ = x_uint8.shape
    dev = x_uint8.device
    x = x_uint8.float()

    def U(a, b):
        return torch.rand(n, device=dev, generator=gen) * (b - a) + a

    def coin(p=0.5):
        return torch.rand(n, device=dev, generator=gen) < p

    # RandomScale(0.07), Rotate(50), Affine(shear=0.1 deg), each p=.5
    scale = torch.where(coin(), U(0.93, 1.07), torch.ones(n, device=dev))
    ang = torch.where(coin(), U(-50, 50), torch.zeros(n, device=dev)) * math.pi / 180
    shear = torch.where(coin(), U(-0.1, 0.1), torch.zeros(n, device=dev)) * math.pi / 180
    # RandomCrop(200) of the scaled image = an output grid covering crop/(s*scale)
    frac = crop / (s * scale)
    max_t = (1 - frac).clamp_min(0)
    tx, ty = U(-1, 1) * max_t, U(-1, 1) * max_t
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = frac * cos
    theta[:, 0, 1] = frac * (-sin + torch.tan(shear) * cos)
    theta[:, 1, 0] = frac * sin
    theta[:, 1, 1] = frac * (cos + torch.tan(shear) * sin)
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, (n, 3, crop, crop), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                      align_corners=False)

    # RandomBrightnessContrast(0.15, 0.1), p=.5, brightness_by_max
    bc = coin()
    alpha = torch.where(bc, U(0.9, 1.1), torch.ones(n, device=dev)).view(n, 1, 1, 1)
    beta = torch.where(bc, U(-0.15, 0.15), torch.zeros(n, device=dev)).view(n, 1, 1, 1) * 255
    x = (x * alpha + beta).clamp(0, 255)

    # Flip(p=.5): horizontal, vertical or both
    fl = coin()
    kind = torch.randint(0, 3, (n,), device=dev, generator=gen)
    h = fl & (kind != 1)
    v = fl & (kind != 0)
    x = torch.where(h.view(n, 1, 1, 1), x.flip(3), x)
    x = torch.where(v.view(n, 1, 1, 1), x.flip(2), x)

    # CoarseDropout(max_holes=8, 16x16, fill 0), p=.5
    cd = coin()
    holes = torch.randint(1, 9, (n,), device=dev, generator=gen)
    yy = torch.arange(crop, device=dev).view(1, 1, crop, 1)
    xx = torch.arange(crop, device=dev).view(1, 1, 1, crop)
    cy = torch.randint(0, crop, (n, 8), device=dev, generator=gen).view(n, 8, 1, 1)
    cx = torch.randint(0, crop, (n, 8), device=dev, generator=gen).view(n, 8, 1, 1)
    hs = torch.randint(1, 17, (n, 8), device=dev, generator=gen).view(n, 8, 1, 1)
    ws = torch.randint(1, 17, (n, 8), device=dev, generator=gen).view(n, 8, 1, 1)
    active = (torch.arange(8, device=dev).view(1, 8) < holes.view(n, 1)) & cd.view(n, 1)
    m = ((yy >= cy) & (yy < cy + hs) & (xx >= cx) & (xx < cx + ws)) & active.view(n, 8, 1, 1)
    x = x * (~m.any(1, keepdim=True)).float()
    return normalize(x)


# --------------------------------------------------------------- the clients
def build_clients_data(cfg, log=print):
    """-> (train_data, test_data, client_center, center_test, truth, meta).

    ``client_center[i]`` is the hospital of client i AND its ground-truth cluster.
    ``truth`` is the planted 2-group labelling when --swap-centers is used, else the
    same as client_center.
    """
    meta, images = build_cache(cfg.data_root, cfg.store_size, log)
    labels_all = meta["target"].values.copy()
    centers_all = meta["center"].values
    fold_all = meta["fold"].values
    swap = [int(c) for c in str(cfg.swap_centers).split(",") if c != ""]
    a, b = (int(v) for v in str(cfg.swap_labels).split(","))
    global_rng = np.random.default_rng(cfg.seed)

    def lab(idx, c):
        y = labels_all[idx].copy()
        if c in swap:                        # same conflict on train AND test
            y = np.where(y == a, b, np.where(y == b, a, y))
        return y

    center_test = []
    for c in range(len(SHORT_NAMES)):
        idx = np.where((centers_all == c) & (fold_all == "test"))[0]
        center_test.append(ClientData(images, idx, lab(idx, c), cfg.batch_size,
                                      cfg.seed, SHORT_NAMES[c] + "_test", False))

    train_data, test_data, client_center, names = [], [], [], []
    for c in cfg.centers_used():
        tr = np.where((centers_all == c) & (fold_all == "train"))[0]
        te = np.where((centers_all == c) & (fold_all == "test"))[0]
        rng = (np.random.default_rng([cfg.seed, c]) if cfg.partition_seed_mode == "per_center"
               else global_rng)
        parts = (partition_pool(tr, cfg.clients_per_center, rng, cfg.partition_mode)
                 if cfg.clients_per_center > 1 else [np.sort(tr)])
        test_parts = (partition_pool(te, cfg.clients_per_center, rng, cfg.partition_mode)
                      if (cfg.split_test_set and cfg.clients_per_center > 1) else None)
        for k, p in enumerate(parts):
            i = len(train_data)
            name = (SHORT_NAMES[c] if cfg.clients_per_center == 1
                    else "{}-{}".format(SHORT_NAMES[c], k))
            train_data.append(ClientData(images, p, lab(p, c), cfg.batch_size,
                                         cfg.seed + 1000 + i, name, True))
            if test_parts is None:
                test_data.append(center_test[c])
            else:
                q = test_parts[k]
                test_data.append(ClientData(images, q, lab(q, c), cfg.batch_size,
                                            cfg.seed + 2000 + i, name + "_test", False))
            client_center.append(c)
            names.append(name)

    client_center = np.array(client_center)
    truth = (np.array([1 if c in swap else 0 for c in client_center])
             if swap else client_center.copy())
    for i, t in enumerate(train_data):
        log("  client {:2d}: {:12s} centre {:10s} | {:5d} train | {:4d} test".format(
            i, names[i], SHORT_NAMES[client_center[i]], len(t), len(test_data[i])))
    return train_data, test_data, client_center, center_test, truth, meta
