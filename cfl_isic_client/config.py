"""Every knob of the 18-client Fed-ISIC2019 CFL study, in one dataclass.

FLamby's values are the defaults where FLamby defines them
(flamby/datasets/fed_isic2019/common.py, flamby/config_isic2019.json); the
CFL-specific ones follow ../cfl-replicate-CIFAR and ../cfl_ixi_client.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class Config:
    # ------------------------------------------------------------------ strategy
    strategy: str = "cfl"            # "cfl" | "fedavg" (never splits) | "partition"
    partition: str = ""              # strategy=partition, e.g. "0,1,2|3,4,5|..."

    # --------------------------------- FLamby Fed-ISIC2019 federated protocol
    communication_rounds: int = 40   # FLamby's own FedAvg budget is get_nb_max_rounds(100) = 9
    local_updates: int = 100         # FLamby benchmark: num_updates per client per round.
                                     # NOTE with 18 clients this is 3x the total work of the
                                     # 6-client setup; --local-updates 33 keeps it equal.
    batch_size: int = 64             # fed_isic2019 BATCH_SIZE
    lr: float = 0.01                 # config_isic2019.json FedAvg learning_rate (SGD)
    momentum: float = 0.0
    lr_decay: float = 1.0
    grad_clip: Optional[float] = 5.0  # NOT in FLamby: without it single clients diverge
                                      # (focal-loss class weights reach 101x)
    weighted_aggregation: bool = True
    bn_stats: str = "local"          # "local" (FLamby strategies aggregate parameters only)
    pretrained: bool = True          # EfficientNet-B0 ImageNet weights (lukemelas)

    # ---------------------------------------------------------- client partition
    clients_per_center: int = 3      # 6 hospitals x 3 = 18 clients
    centers: str = "0,1,2,3,4,5"
    partition_mode: str = "array_split"      # | "equal_drop" (CIFAR arithmetic)
    partition_seed_mode: str = "per_center"  # | "global"
    split_test_set: bool = False     # sub-clients share their centre's test set

    # ---------------------- label-conflict injection (optional ground truth) ----
    swap_centers: str = ""           # e.g. "3,4,5": these CENTRES swap two labels,
                                     # train and test, giving a planted 2-cluster truth
    swap_labels: str = "0,1"         # MEL <-> NV by default

    # ------------------------------------------- CFL criteria (Alg. 5, l. 14-20)
    eps_1: float = 0.26              # ||weighted mean dW|| below this  (Eq. 32)
    eps_2: float = 0.74              # max_i ||dW_i||    above this     (Eq. 33)
    weighted_mean_norm: bool = True
    min_split_round: int = 0         # 0 = paper (no extra guard)
    min_cluster_size: int = 1
    split_cooldown: int = 0          # 0 = paper (no extra guard)
    max_clusters: Optional[int] = None
    split_schedule: tuple = ()
    gamma_max: Optional[float] = None
    gamma_auto: bool = False
    gamma_probe_every: int = 0
    gamma_probe_client: int = 0

    # ---------------------------------------------------------------- stragglers
    straggler_mode: str = "none"     # "none" | "partial" | "drop"
    straggler_profile: str = "tiers"
    straggler_speeds: str = ""
    straggler_tiers: str = "1.0,0.5,0.25"   # cycled over sub-clients inside each centre
    straggler_sigma: float = 0.3
    straggler_dynamic: bool = False
    straggler_deadline: float = 1.0
    straggler_min_updates: int = 1
    straggler_seed: int = 7

    # ------------------------------------------------------------ eval / data / io
    eval_every: int = 5              # 18 clients x their centre test set = 13 950 images,
                                     # so evaluating every round would cost ~80 s/round
    cross_eval_every: int = 10
    data_root: str = "isic"          # holds train_test_split.csv + the preprocessed jpgs
    store_size: int = 220            # cached centre crop (>= the 200 px train/test crop)
    eval_batch_size: int = 256
    amp: bool = True                 # bf16 autocast; weights and updates stay fp32
    deterministic: bool = False
    device: str = "auto"
    seed: int = 42
    run_name: str = "cfl18"
    results_dir: str = "results_isic18"
    save_similarities: bool = True
    save_models: bool = False        # EfficientNet is 16 MB per client
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
