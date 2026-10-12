"""Every knob of the 9-client Fed-IXI CFL study, in one dataclass.

Defaults follow FLamby's Fed-IXI benchmark (flamby/datasets/fed_ixi/common.py,
flamby/config_ixi.json) where FLamby defines a value, and the CIFAR replication
(../cfl-replicate-CIFAR) for the CFL-specific ones.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class Config:
    # ------------------------------------------------------------------ strategy
    strategy: str = "cfl"            # "cfl" | "fedavg" (never splits) | "partition" (fixed)
    partition: str = ""              # strategy=partition, e.g. "0,1,2|3,4,5|6,7,8"

    # ------------------------------------- FLamby Fed-IXI federated protocol
    communication_rounds: int = 60
    local_updates: int = 100         # FLamby benchmark: num_updates per round
    batch_size: int = 2              # fed_ixi BATCH_SIZE
    lr: float = 1e-3                 # fed_ixi LR (FedAvg uses SGD at this lr)
    momentum: float = 0.0
    lr_decay: float = 1.0            # per round
    grad_clip: Optional[float] = None
    weighted_aggregation: bool = True   # FedAvg weights clients by |D_i|
    bn_stats: str = "local"          # "local" (FLamby/CFL: params only) | "average"

    # ---------------------------------------------------------- client partition
    clients_per_center: int = 3      # 3 hospitals x 3 = 9 clients
    centers: str = "0,1,2"           # Guys=0, HH=1, IOP=2
    partition_mode: str = "array_split"   # "array_split" (keep all scans)
                                          # | "equal_drop" (CIFAR: equal blocks, drop remainder)
    partition_seed_mode: str = "per_center"  # "per_center" (rng seeded [seed, center])
                                             # | "global" (one CIFAR-style stream)
    split_test_set: bool = False     # False: sub-clients share their centre's test set

    # ------------------------------------------- CFL criteria (Alg. 5, l. 14-20)
    eps_1: float = 0.0095            # ||weighted mean dW|| must be BELOW this (Eq. 32)
    eps_2: float = 0.014             # max_i ||dW_i|| must be ABOVE this      (Eq. 33)
    weighted_mean_norm: bool = True  # Eq. 32 weights by |D_i| / |D_c|
    min_split_round: int = 0         # 0 = paper (no extra guard)
    min_cluster_size: int = 1        # split any cluster with > 1 client
    split_cooldown: int = 0          # 0 = paper (no extra guard)
    max_clusters: Optional[int] = None
    split_schedule: tuple = ()       # force splits at these rounds, ignore eps
    gamma_max: Optional[float] = None     # Alg. 5 line 18; None = check disabled
    gamma_auto: bool = False         # gamma_max := median of last 3 gamma probes
    gamma_probe_every: int = 0       # every k rounds: train 2 i.i.d. halves of one client
    gamma_probe_client: int = 0      # which client to probe (0 = first Guys sub-client)

    # ---------------------------------------------------------------- stragglers
    straggler_mode: str = "none"     # "none" | "partial" (fewer local steps)
                                     # | "drop" (client misses the round entirely)
    straggler_profile: str = "tiers"  # "tiers" | "lognormal" | "uniform" | "manual"
    straggler_speeds: str = ""       # manual: comma list, one relative speed per client
    straggler_tiers: str = "1.0,0.5,0.25"   # cycled over sub-clients within each centre
    straggler_sigma: float = 0.3     # lognormal sigma / uniform half-width
    straggler_dynamic: bool = False  # True: resample speeds every round
    straggler_deadline: float = 1.0  # round budget, in units of local_updates at speed 1
    straggler_min_updates: int = 1   # a "partial" straggler always does at least this many
    straggler_seed: int = 7          # separate from the data seed

    # ----------------------------------------------------------------- the model
    # FLamby's Fed-IXI 3D U-Net (246 156 parameters), built from the pytorch-3dunet
    # blocks; see models.py.  The four fields below override the preset for ablations:
    # 0 / "" means "keep FLamby's value".
    f_maps: int = 0                  # FLamby: 16  -> block widths (16, 32, 64)
    num_levels: int = 0              # FLamby: 3
    layer_order: str = ""            # FLamby: "cbp" = conv + BatchNorm + PReLU
    upsample: str = ""               # FLamby: "trilinear" (upsampling_type="linear")
    dropout_prob: float = 0.0

    # ------------------------------------------------------------ eval / data / io
    eval_every: int = 1
    cross_eval_every: int = 10       # every client model x every centre test set
    data_root: str = "ixi/7kd5wj7v7p-3/IXI_sample"
    cache_dir: str = "cache"
    deterministic: bool = True
    device: str = "auto"             # "auto" | "cuda" | "cpu"
    pin_data_to_device: bool = True  # keep all 566 volumes in VRAM (~0.4 GB)
    seed: int = 42
    run_name: str = "cfl9"
    results_dir: str = "results_ixi9"
    save_similarities: bool = True
    save_models: bool = True
    torch_threads: int = 8
    log_every: int = 1

    # ---------------------------------------------------------------- helpers
    def centers_used(self):
        return [int(c) for c in str(self.centers).split(",") if c != ""]

    def n_clients(self):
        return len(self.centers_used()) * self.clients_per_center

    def to_json(self, path):
        d = asdict(self)
        d["split_schedule"] = list(d["split_schedule"])
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        return d
