"""Fed-IXI (IXI Tiny) loading, FLamby preprocessing, and the 9-client partition.

FLamby pipeline reproduced here (flamby/datasets/fed_ixi/dataset.py + utils.py):
  * modality T1 only (in the Mendeley IXI_sample release T2 is a byte copy of T1);
  * image: ToTensor -> AddChannel -> monai Resize((48, 60, 48)) [mode 'area']
           -> monai NormalizeIntensity (whole-volume z-score);
  * label: same Resize -> AsDiscrete(to_onehot=2).  MONAI's one_hot casts with
           ``.long()``, i.e. TRUNCATES the area-resized label, so only voxels whose
           whole receptive window is brain stay foreground.  Kept as-is;
  * centres parsed from the folder name, {"Guys": 0, "HH": 1, "IOP": 2};
  * train/test: per centre round(0.8 n) train, shuffled with np.random.seed(10),
    drawn in the order HH, Guys, IOP (_create_train_test_split).  The per-centre
    counts match FLamby's README exactly: Guys 249/62, HH 145/36, IOP 59/15.

THE 9-CLIENT PARTITION (what differs from plain Fed-IXI)
--------------------------------------------------------
Each centre's TRAIN set is divided into ``clients_per_center`` sub-clients with the
recipe of ../cfl-replicate-CIFAR (cfl_cifar/data_utils.py::partition):

    idcs = rng.permutation(n)           # one seeded shuffle
    blocks = contiguous slices of idcs  # disjoint, (near-)equal, unstratified

applied ONCE PER CENTRE, so a sub-client only ever holds scans of its own hospital.
Client i belongs to centre ``i // clients_per_center``, and that centre is the
GROUND-TRUTH CLUSTER against which the recovered partition is scored (ARI, gap).

  Guys 249 -> 83 / 83 / 83   -> clients 0,1,2
  HH   145 -> 49 / 48 / 48   -> clients 3,4,5
  IOP   59 -> 20 / 20 / 19   -> clients 6,7,8

``partition_mode="equal_drop"`` reproduces CIFAR's exact arithmetic (``per = n // k``
equal blocks, remainder discarded: 83/83/83, 48/48/48, 19/19/19).
``partition_seed_mode="per_center"`` seeds each centre's shuffle with ``[seed, c]`` so
changing one centre's client count leaves the others untouched; ``"global"`` uses a
single CIFAR-style stream across centres.

Preprocessed volumes are cached as .npz (~0.4 GB) so later runs start in seconds.
"""
import csv
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

CENTERS = ["Guys", "HH", "IOP"]
CENTER_ID = {c: i for i, c in enumerate(CENTERS)}
COMMON_SHAPE = (48, 60, 48)
CACHE_NAME = "ixi_tiny_T1_48x60x48.npz"


# ------------------------------------------------------------------ subjects
def list_subjects(root):
    """-> [{name, patient_id, center}] for every subject folder, sorted by name."""
    subjects = []
    for d in sorted(glob.glob(os.path.join(root, "IXI*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        subjects.append({"name": name, "patient_id": int(name[3:6]),
                         "center": name.split("-")[1]})
    if not subjects:
        raise FileNotFoundError("no IXI* subject folders under " + os.path.abspath(root))
    return subjects


def flamby_split(centers):
    """Replicates flamby.datasets.fed_ixi.utils._create_train_test_split."""
    rs = np.random.RandomState(10)          # == np.random.seed(10) + np.random.shuffle
    split = {}
    for c in ["HH", "Guys", "IOP"]:         # FLamby's shuffle order drives the RNG
        n = centers.count(c)
        n_train = round(n * 0.8)
        s = ["train"] * n_train + ["test"] * (n - n_train)
        rs.shuffle(s)
        split[c] = s
    out, pos = [], {c: 0 for c in split}
    for c in centers:
        out.append(split[c][pos[c]])
        pos[c] += 1
    return out


# ------------------------------------------------------------- preprocessing
def _load_nii(path):
    import nibabel as nib
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float64)


def preprocess_subject(folder):
    """-> (image float32 (48,60,48), label uint8 (48,60,48), raw-stats dict)."""
    img = _load_nii(glob.glob(os.path.join(folder, "T1", "*.nii.gz"))[0])
    lab = _load_nii(glob.glob(os.path.join(folder, "label", "*.nii.gz"))[0])
    brain = lab > 0.5
    stats = {"raw_shape": "x".join(map(str, img.shape)),
             "raw_max": float(img.max()),
             "raw_p99": float(np.percentile(img, 99)),
             "raw_brain_mean": float(img[brain].mean()),
             "raw_brain_std": float(img[brain].std()),
             "raw_bg_mean": float(img[~brain].mean()),
             "raw_brain_fraction": float(brain.mean())}

    t = torch.from_numpy(img)[None, None]
    t = F.interpolate(t, size=COMMON_SHAPE, mode="area")[0, 0]
    std = t.std()
    t = (t - t.mean()) / (std if std > 0 else 1.0)              # NormalizeIntensity

    lb = torch.from_numpy(lab)[None, None]
    lb = F.interpolate(lb, size=COMMON_SHAPE, mode="area")[0, 0].long()  # one_hot truncation
    stats["brain_fraction_48"] = float(lb.float().mean())
    return t.float().numpy(), lb.to(torch.uint8).numpy(), stats


def load_ixi(root, cache_dir, log=print):
    """-> dict(x (N,1,48,60,48) float32, y (N,48,60,48) uint8, meta list[dict])."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, CACHE_NAME)
    meta_csv = os.path.join(cache_dir, "subjects.csv")
    subjects = list_subjects(root)
    if os.path.exists(cache) and os.path.exists(meta_csv):
        z = np.load(cache)
        if len(z["names"]) == len(subjects):
            with open(meta_csv, newline="") as f:
                return {"x": z["x"], "y": z["y"], "meta": list(csv.DictReader(f))}
    log("preprocessing {} subjects -> {} (one-off, ~1-2 min)".format(len(subjects), cache))
    splits = flamby_split([s["center"] for s in subjects])
    xs, ys, meta = [], [], []
    for s, sp in zip(subjects, splits):
        x, y, st = preprocess_subject(os.path.join(root, s["name"]))
        xs.append(x[None])
        ys.append(y)
        meta.append({**s, "center_id": CENTER_ID[s["center"]], "split": sp, **st})
    x, y = np.stack(xs), np.stack(ys)
    np.savez(cache, x=x, y=y, names=np.array([m["name"] for m in meta]))
    with open(meta_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    with open(meta_csv, newline="") as f:        # re-read so dtypes match the cached path
        meta = list(csv.DictReader(f))
    return {"x": x, "y": y, "meta": meta}


# --------------------------------------------------------- the CIFAR-style split
def partition_pool(idx, n_parts, rng, mode="array_split"):
    """One seeded shuffle, then contiguous blocks (cfl_cifar/data_utils.py::partition).

    array_split : near-equal blocks, every scan used (83/83/83, 49/48/48, 20/20/19)
    equal_drop  : CIFAR arithmetic ``per = n // n_parts``; the remainder is dropped
    """
    idx = np.asarray(idx)
    shuffled = idx[rng.permutation(len(idx))]      # shuffle, then slice contiguously
    if mode == "equal_drop":
        per = len(idx) // n_parts
        return [np.sort(shuffled[i * per:(i + 1) * per]) for i in range(n_parts)]
    if mode != "array_split":
        raise ValueError("partition_mode must be 'array_split' or 'equal_drop'")
    return [np.sort(p) for p in np.array_split(shuffled, n_parts)]


# ------------------------------------------------------------ client datasets
class TensorClientData:
    """In-memory client dataset with FLamby's ``DataLoaderWithMemory`` semantics:
    ``next_batch()`` walks shuffled epochs forever, so a round of ``num_updates``
    steps may straddle an epoch boundary."""

    def __init__(self, x, y, batch_size, seed, device=None, name="", indices=None):
        if device is not None:
            x, y = x.to(device), y.to(device)
        self.x, self.y, self.batch_size, self.name = x, y, batch_size, name
        # dataset-level indices of these scans, kept so a run can be replayed exactly
        self.indices = None if indices is None else np.asarray(indices).tolist()
        self.g = torch.Generator().manual_seed(int(seed))   # CPU gen: same order on CPU/GPU
        self._order, self._pos = None, 0

    def __len__(self):
        return len(self.y)

    def next_batch(self):
        if self._order is None or self._pos >= len(self._order):
            self._order, self._pos = torch.randperm(len(self.y), generator=self.g), 0
        j = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        if self.x.device.type != "cpu":
            j = j.to(self.x.device)
        return self.x[j], self.y[j]

    def batches(self, shuffle=True, batch_size=None):
        bs = batch_size or self.batch_size
        n = len(self.y)
        order = torch.randperm(n, generator=self.g) if shuffle else torch.arange(n)
        if self.x.device.type != "cpu":
            order = order.to(self.x.device)
        for i in range(0, n, bs):
            j = order[i:i + bs]
            yield self.x[j], self.y[j]


def build_clients_data(cfg, device=None, log=print):
    """-> (train_data, test_data, client_center, center_test, meta).

    ``client_center[i]`` is the hospital of client i AND its ground-truth cluster.
    Clients are numbered centre-major: with clients_per_center=3, 0-2 = Guys,
    3-5 = HH, 6-8 = IOP.
    """
    d = load_ixi(cfg.data_root, cfg.cache_dir, log)
    x, y, meta = torch.from_numpy(d["x"]), torch.from_numpy(d["y"]), d["meta"]
    centers = np.array([int(m["center_id"]) for m in meta])
    is_train = np.array([m["split"] == "train" for m in meta])
    dev = device if cfg.pin_data_to_device else None
    global_rng = np.random.default_rng(cfg.seed)     # used when partition_seed_mode="global"

    center_test = []
    for c in range(len(CENTERS)):
        idx = torch.from_numpy(np.where((centers == c) & ~is_train)[0])
        center_test.append(TensorClientData(x[idx], y[idx], cfg.batch_size, cfg.seed,
                                            dev, CENTERS[c] + "_test",
                                            indices=idx.numpy()))

    train_data, test_data, client_center, client_names = [], [], [], []
    for c in cfg.centers_used():
        tr_idx = np.where((centers == c) & is_train)[0]
        te_idx = np.where((centers == c) & ~is_train)[0]
        rng = (np.random.default_rng([cfg.seed, c]) if cfg.partition_seed_mode == "per_center"
               else global_rng)
        parts = (partition_pool(tr_idx, cfg.clients_per_center, rng, cfg.partition_mode)
                 if cfg.clients_per_center > 1 else [np.sort(tr_idx)])
        test_parts = None
        if cfg.split_test_set and cfg.clients_per_center > 1:
            test_parts = partition_pool(te_idx, cfg.clients_per_center, rng, cfg.partition_mode)
        for k, p in enumerate(parts):
            i = len(train_data)
            name = CENTERS[c] if cfg.clients_per_center == 1 else "{}-{}".format(CENTERS[c], k)
            train_data.append(TensorClientData(x[torch.from_numpy(p)], y[torch.from_numpy(p)],
                                               cfg.batch_size, cfg.seed + 1000 + i, dev, name,
                                               indices=p))
            if test_parts is None:
                test_data.append(center_test[c])      # sub-clients share the centre test set
            else:
                q = torch.from_numpy(test_parts[k])
                test_data.append(TensorClientData(x[q], y[q], cfg.batch_size,
                                                  cfg.seed + 2000 + i, dev, name + "_test",
                                                  indices=test_parts[k]))
            client_center.append(c)
            client_names.append(name)

    for i, t in enumerate(train_data):
        log("  client {}: {:10s} centre {:4s} | {:3d} train | {:3d} test".format(
            i, client_names[i], CENTERS[client_center[i]], len(t), len(test_data[i])))
    return train_data, test_data, np.array(client_center), center_test, meta
