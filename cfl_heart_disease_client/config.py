"""Every knob of the 18-client Fed-Heart-Disease CFL study.

FLamby's values are the defaults where FLamby defines them
(flamby/datasets/fed_heart_disease/common.py, flamby/config_heart_disease.json).
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class Config:
    # ------------------------------------------------------------------ strategy
    strategy: str = "cfl"            # "cfl" | "fedavg" | "partition"
    partition: str = ""              # e.g. "0,1,2|3,4,5|..."

    # ------------------------------- FLamby Fed-Heart-Disease federated protocol
    communication_rounds: int = 100  # FLamby's own budget: get_nb_max_rounds(100) = 15
    local_updates: int = 20          # steps per client per round.  With 18 clients the
                                     # smallest holds 13 patients = 2 batches, so 20
                                     # steps is ~10 passes over it per round.
    batch_size: int = 4              # fed_heart_disease BATCH_SIZE
    lr: float = 0.001                # config_heart_disease.json FedAvg (SGD)
    momentum: float = 0.0
    lr_decay: float = 1.0
    grad_clip: Optional[float] = None
    weighted_aggregation: bool = True
    bn_stats: str = "local"          # the model has no BatchNorm; kept for symmetry

    # ---------------------------------------------------------- client partition
    clients_per_center: int = 2      # 4 hospitals x 2 = 8 clients
    centers: str = "0,1,2,3"
    partition_mode: str = "array_split"      # | "equal_drop" (CIFAR arithmetic)
    partition_seed_mode: str = "per_center"  # | "global"
    partition_stratify: bool = True  # shuffle-and-split WITHIN the event/censored groups,
                                     # so each sub-client gets its share of the deaths.
                                     # False = the plain CIFAR recipe (see data_utils).
    split_test_set: bool = False     # sub-clients share their region's test set

    # ------------------------------------------- CFL criteria (Alg. 5, l. 14-20)
    eps_1: float = 0.002
    eps_2: float = 0.005
    weighted_mean_norm: bool = True
    min_split_round: int = 0
    min_cluster_size: int = 1
    split_cooldown: int = 0
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
    straggler_tiers: str = "1.0,0.5,0.25"
    straggler_sigma: float = 0.3
    straggler_dynamic: bool = False
    straggler_deadline: float = 1.0
    straggler_min_updates: int = 1
    straggler_seed: int = 7

    # ------------------------------------------------------------ eval / data / io
    eval_every: int = 1
    cross_eval_every: int = 10
    data_root: str = "heart"          # the four UCI processed.*.data files
    deterministic: bool = True
    device: str = "auto"
    seed: int = 42
    run_name: str = "cfl8"
    results_dir: str = "results_heart8"
    save_similarities: bool = True
    save_models: bool = True         # 40 floats per client
    torch_threads: int = 8
    log_every: int = 10

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
