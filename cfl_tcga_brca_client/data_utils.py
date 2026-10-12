"""Fed-TCGA-BRCA data: FLamby's files and split, plus the 18-client partition.

FLamby (flamby/datasets/fed_tcga_brca/dataset.py) ships the data inside the repo:

    brca.csv                 1096 rows: pid, 39 clinical covariates, E (event), T (time)
    dataset_creation_scripts/train_test_split.csv
                             pid -> fold (train/test) and fold2 (train_<c>/test_<c>)
    centers.csv              TSS code -> source site -> region (the 6 centres)

``__getitem__`` returns ``x = row[1:40]`` (39 features, already one-hot encoded and
scaled by FLamby's own preprocessing) and ``y = row[40:42] = (E, T)``.  No further
normalisation is applied -- unlike Fed-Heart-Disease, the published csv is ready to use.

The 6 centres are geographic regions of the TCGA source sites:

    0  311 patients (248 train / 63 test)      3  162 (129 / 33)
    1  196           (156 / 40)                4  162 (129 / 33)
    2  206           (164 / 42)                5   51 ( 40 / 11)

Survival task: only 13.8 % of patients have an observed event (death); the rest are
right-censored, which the Cox loss and the c-index both account for.

THE 18-CLIENT PARTITION
-----------------------
Each centre's TRAIN set is divided into ``clients_per_center`` sub-clients with the
recipe of ../../cfl-replicate-CIFAR (cfl_cifar/data_utils.py::partition):

    idcs = rng.permutation(n)           # one seeded shuffle
    blocks = contiguous slices of idcs  # disjoint, (near-)equal, unstratified

applied ONCE PER CENTRE, so a sub-client only ever holds patients of its own region,
and the region is the GROUND-TRUTH CLUSTER used to score the recovered partition.

    centre 0: 248 -> 83 / 83 / 82      centre 3: 129 -> 43 / 43 / 43
    centre 1: 156 -> 52 / 52 / 52      centre 4: 129 -> 43 / 43 / 43
    centre 2: 164 -> 55 / 55 / 54      centre 5:  40 -> 14 / 13 / 13
"""
import os

import numpy as np
import pandas as pd
import torch

N_FEATURES = 39
N_CENTERS = 6


def partition_pool(idx, n_parts, rng, mode="array_split", strata=None):
    """One seeded shuffle, then contiguous blocks (cfl_cifar/data_utils.py::partition).

    ``strata`` (optional) applies that same recipe INSIDE each stratum and concatenates
    the results, so every sub-client gets a proportional share of each stratum.  For
    survival data the stratum is the event indicator E: without it the Cox partial
    likelihood of a sub-client that happens to receive no deaths is identically zero,
    its update is the zero vector, and its cosine similarity with every other client is
    undefined (0/0).  That is not a hypothetical -- with 3 sub-clients per centre and
    seed 42, client C5-1 draws 0 of the 2 events in its region.
    """
    idx = np.asarray(idx)
    if strata is not None:
        strata = np.asarray(strata)
        parts = [[] for _ in range(n_parts)]
        for level in np.unique(strata):
            sub = idx[strata == level]
            for k, blk in enumerate(partition_pool(sub, n_parts, rng, mode)):
                parts[k].append(blk)
        return [np.sort(np.concatenate(p)) if len(p) else np.array([], dtype=int)
                for p in parts]
    shuffled = idx[rng.permutation(len(idx))]
    if mode == "equal_drop":
        per = len(idx) // n_parts
        return [np.sort(shuffled[i * per:(i + 1) * per]) for i in range(n_parts)]
    if mode != "array_split":
        raise ValueError("partition_mode must be 'array_split' or 'equal_drop'")
    return [np.sort(p) for p in np.array_split(shuffled, n_parts)]


class TensorClientData:
    """In-memory client dataset with FLamby's DataLoaderWithMemory semantics."""

    def __init__(self, x, y, batch_size, seed, name="", indices=None, device=None):
        if device is not None:
            x, y = x.to(device), y.to(device)
        self.x, self.y, self.batch_size, self.name = x, y, batch_size, name
        self.indices = None if indices is None else np.asarray(indices).tolist()
        self.g = torch.Generator().manual_seed(int(seed))
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

    def batches(self, shuffle=False, batch_size=None):
        bs = batch_size or len(self.y)          # the sets are tiny: one batch by default
        n = len(self.y)
        order = torch.randperm(n, generator=self.g) if shuffle else torch.arange(n)
        if self.x.device.type != "cpu":
            order = order.to(self.x.device)
        for i in range(0, n, bs):
            j = order[i:i + bs]
            yield self.x[j], self.y[j]

    def all(self):
        return self.x, self.y


def load_brca(root, log=print):
    """-> (X (N,39) float32, Y (N,2) float32 = (E,T), center (N,), fold (N,) of str)."""
    data = pd.read_csv(os.path.join(root, "brca.csv"))
    split = pd.read_csv(os.path.join(root, "train_test_split.csv"))
    df = data.merge(split, on="pid", how="inner")        # 1088 of the 1096 rows are split
    x = df.iloc[:, 1:1 + N_FEATURES].values.astype(np.float32)
    y = df.iloc[:, 1 + N_FEATURES:3 + N_FEATURES].values.astype(np.float32)   # E, T
    center = df["fold2"].str.split("_").str[1].astype(int).values
    fold = df["fold"].values
    log("brca.csv: {} patients, {} features, event rate {:.3f}".format(
        len(df), x.shape[1], y[:, 0].mean()))
    return torch.from_numpy(x), torch.from_numpy(y), center, fold


def build_clients_data(cfg, device=None, log=print):
    """-> (train_data, test_data, client_center, center_test, meta)."""
    x, y, center, fold = load_brca(cfg.data_root, log)
    is_train = fold == "train"
    global_rng = np.random.default_rng(cfg.seed)

    center_test = []
    for c in range(N_CENTERS):
        idx = np.where((center == c) & ~is_train)[0]
        center_test.append(TensorClientData(x[idx], y[idx], cfg.batch_size, cfg.seed,
                                            "C{}_test".format(c), idx, device))

    train_data, test_data, client_center, names = [], [], [], []
    for c in cfg.centers_used():
        tr = np.where((center == c) & is_train)[0]
        te = np.where((center == c) & ~is_train)[0]
        rng = (np.random.default_rng([cfg.seed, c]) if cfg.partition_seed_mode == "per_center"
               else global_rng)
        strata = y[tr, 0].numpy() if cfg.partition_stratify else None
        parts = (partition_pool(tr, cfg.clients_per_center, rng, cfg.partition_mode, strata)
                 if cfg.clients_per_center > 1 else [np.sort(tr)])
        test_parts = (partition_pool(te, cfg.clients_per_center, rng, cfg.partition_mode)
                      if (cfg.split_test_set and cfg.clients_per_center > 1) else None)
        for k, p in enumerate(parts):
            i = len(train_data)
            name = ("C{}".format(c) if cfg.clients_per_center == 1
                    else "C{}-{}".format(c, k))
            train_data.append(TensorClientData(x[p], y[p], cfg.batch_size,
                                               cfg.seed + 1000 + i, name, p, device))
            if test_parts is None:
                test_data.append(center_test[c])
            else:
                q = test_parts[k]
                test_data.append(TensorClientData(x[q], y[q], cfg.batch_size,
                                                  cfg.seed + 2000 + i, name + "_test",
                                                  q, device))
            client_center.append(c)
            names.append(name)

    client_center = np.array(client_center)
    starved = [names[i] for i, d in enumerate(train_data) if float(d.y[:, 0].sum()) < 2]
    if starved:
        log("  WARNING: {} client(s) hold fewer than 2 events ({}); their Cox gradient is"
            " near-zero, so their update direction carries little information"
            .format(len(starved), ", ".join(starved)))
    for i, d in enumerate(train_data):
        log("  client {:2d}: {:7s} centre {} | {:4d} train ({:2.0f} events) | {:3d} test".format(
            i, names[i], client_center[i], len(d), float(d.y[:, 0].sum()),
            len(test_data[i])))
    return train_data, test_data, client_center, center_test, {"names": names}
