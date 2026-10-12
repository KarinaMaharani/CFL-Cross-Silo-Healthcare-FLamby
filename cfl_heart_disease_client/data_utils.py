"""Fed-Heart-Disease data: FLamby's preprocessing and split, plus the 8-client partition.

FLamby (flamby/datasets/fed_heart_disease/dataset.py) reads the four UCI Heart Disease
files -- one per hospital -- and does:

  1. ``df.replace("?", NaN).drop([10, 11, 12], axis=1).dropna(axis=0)``
     (columns 10-12 are slope / ca / thal, missing for most non-Cleveland patients);
  2. the last column is the label, binarised: 0 stays 0, anything > 0 becomes 1
     ("some degree of heart disease");
  3. ``pd.get_dummies(features, columns=[2, 6], drop_first=True)`` on chest-pain type
     and resting ECG, which turns 11 raw columns into **13 features**;
  4. a per-centre train/test split with ``train_fraction = 0.66``,
     ``train_test_split(..., random_state=43, shuffle=True, stratify=binarised label)``
     (stratification is skipped when a centre has one class, or fewer than 3 of a class);
  5. **per-centre normalisation**: each centre's features are standardised with the mean
     and std of that centre's own TRAINING patients.  FLamby computes pooled-train
     statistics separately for the pooled baseline; both are provided here.

The four centres (and why they differ):

    0 Cleveland    1 Hungary    2 Switzerland    3 Long Beach VA

Switzerland is the interesting one: nearly every patient there is positive, so its
conditional P(y|x) is close to constant. That is label-distribution conflict of the
kind CFL is meant to detect, and it is why FLamby replaced ROC-AUC with accuracy.

THE 8-CLIENT PARTITION
----------------------
Each centre's TRAIN set is divided into ``clients_per_center`` sub-clients with the
recipe of ../../cfl-replicate-CIFAR (cfl_cifar/data_utils.py::partition):

    idcs = rng.permutation(n)           # one seeded shuffle
    blocks = contiguous slices of idcs  # disjoint, (near-)equal, unstratified

applied ONCE PER CENTRE, so a sub-client only ever holds patients of its own hospital,
and the hospital is the GROUND-TRUTH CLUSTER used to score the recovered partition.
``partition_stratify`` (default on) runs that same recipe inside the positive and
negative groups, so each sub-client keeps its centre's class balance -- with as few as
15 patients per client, an unstratified draw can otherwise hand a client a single class
and a degenerate BCE gradient.
"""
import os

import numpy as np
import pandas as pd
import torch

CENTER_FILES = {"cleveland": 0, "hungarian": 1, "switzerland": 2, "va": 3}
CENTER_NAMES = ["Cleveland", "Hungary", "Switzerland", "LongBeachVA"]
N_FEATURES = 13
TRAIN_FRACTION = 0.66


def partition_pool(idx, n_parts, rng, mode="array_split", strata=None):
    """One seeded shuffle, then contiguous blocks (cfl_cifar/data_utils.py::partition).

    With ``strata`` the same recipe is applied inside each stratum and the blocks are
    concatenated, so every sub-client keeps the centre's class balance.
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
        bs = batch_size or len(self.y)
        n = len(self.y)
        order = torch.randperm(n, generator=self.g) if shuffle else torch.arange(n)
        if self.x.device.type != "cpu":
            order = order.to(self.x.device)
        for i in range(0, n, bs):
            j = order[i:i + bs]
            yield self.x[j], self.y[j]

    def all(self):
        return self.x, self.y


def load_heart(root, log=print):
    """FLamby's HeartDiseaseRaw preprocessing.

    -> (X (N,13) float32 normalised per centre, y (N,1) float32, center (N,), is_train (N,))
    """
    from sklearn.model_selection import train_test_split

    frames, labels, centers, sets = [], [], [], []
    files = sorted(f for f in os.listdir(root) if f.endswith(".data"))
    if not files:
        raise FileNotFoundError(
            "no processed.*.data files in {}. Download them first:\n"
            "  python -m cfl_heart_disease_client.download".format(os.path.abspath(root)))
    for fname in files:
        center_name = fname.split(".")[1]
        df = pd.read_csv(os.path.join(root, fname), header=None)
        df = df.replace("?", np.nan).drop([10, 11, 12], axis=1).dropna(axis=0)
        df = df.apply(pd.to_numeric)
        frames.append(df.iloc[:, :-1])
        labels.append(df.iloc[:, -1])
        n = df.shape[0]
        centers += [CENTER_FILES[center_name]] * n

        binary = df.iloc[:, -1].where(df.iloc[:, -1] == 0, 1)
        levels = np.unique(binary)
        stratify = binary if (len(levels) > 1 and all((binary == l).sum() > 2 for l in levels)) \
            else None
        idx_train, idx_test = train_test_split(
            np.arange(n), test_size=1.0 - TRAIN_FRACTION, train_size=TRAIN_FRACTION,
            random_state=43, shuffle=True, stratify=stratify)
        sets += ["test" if i in set(idx_test.tolist()) else "train" for i in range(n)]
        log("  {:12s} {:3d} patients ({} train / {} test), positive rate {:.2f}".format(
            center_name, n, len(idx_train), len(idx_test), float(binary.mean())))

    features = pd.concat(frames, ignore_index=True)
    features = pd.get_dummies(features, columns=[2, 6], drop_first=True)
    x = torch.from_numpy(features.values.astype(np.float32))
    y = pd.concat(labels, ignore_index=True)
    y = torch.from_numpy(y.where(y == 0, 1).values.astype(np.float32)).unsqueeze(1)
    center = np.array(centers)
    is_train = np.array([s == "train" for s in sets])
    if x.shape[1] != N_FEATURES:
        raise ValueError("expected {} features after get_dummies, got {}".format(
            N_FEATURES, x.shape[1]))

    # per-centre standardisation, statistics from that centre's TRAIN patients only
    x_norm = x.clone()
    for c in range(len(CENTER_NAMES)):
        tr = (center == c) & is_train
        if tr.sum() == 0:
            continue
        mean, std = x[tr].mean(0), x[tr].std(0)
        std = torch.where(std > 0, std, torch.ones_like(std))
        x_norm[center == c] = (x[center == c] - mean) / std
    return x_norm, y, center, is_train


def build_clients_data(cfg, device=None, log=print):
    """-> (train_data, test_data, client_center, center_test, meta)."""
    x, y, center, is_train = load_heart(cfg.data_root, log)
    global_rng = np.random.default_rng(cfg.seed)

    center_test = []
    for c in range(len(CENTER_NAMES)):
        idx = np.where((center == c) & ~is_train)[0]
        center_test.append(TensorClientData(x[idx], y[idx], cfg.batch_size, cfg.seed,
                                            CENTER_NAMES[c] + "_test", idx, device))

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
            name = (CENTER_NAMES[c] if cfg.clients_per_center == 1
                    else "{}-{}".format(CENTER_NAMES[c], k))
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
    for i, d in enumerate(train_data):
        log("  client {}: {:15s} centre {:12s} | {:3d} train (pos {:.2f}) | {:3d} test".format(
            i, names[i], CENTER_NAMES[client_center[i]], len(d), float(d.y.mean()),
            len(test_data[i])))
    return train_data, test_data, client_center, center_test, {"names": names}
