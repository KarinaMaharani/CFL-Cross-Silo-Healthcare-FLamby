# cfl_heart_disease_client

Clustered Federated Learning (CFL) on the Fed-Heart-Disease benchmark from FLamby, with 8 sub-clients drawn from 4 hospitals. The model is a 14-parameter logistic regression, so the whole study runs in a few minutes on a CPU.

## Quick start

```bash
python -m cfl_heart_disease_client.download      # once: ~40 KB from UCI into heart/
bash cfl_heart_disease_client/run_all.sh          # the full study
```

Results are written to `results_heart8/` in the project root.

## Files

| File | What it does |
|---|---|
| `__init__.py` | Empty. Marks the folder as a Python package so the `python -m cfl_heart_disease_client.<module>` commands work. |
| `download.py` | Downloads the four UCI Heart Disease files (Cleveland, Hungary, Switzerland, VA) that FLamby uses into `heart/`. Skips files that already exist. |
| `data_utils.py` | Loads and preprocesses the four files the way FLamby does (drops columns 10–12, binarises the label, one-hot encodes two columns to get 13 features, 66/34 train/test split per hospital, per-hospital standardisation). Splits each hospital's training set into `clients_per_center` sub-clients (2 by default, so 8 in total), stratified by label. |
| `models.py` | The FLamby baseline: `Linear(13, 1)` followed by a sigmoid, binary cross-entropy loss, and accuracy as the metric. |
| `config.py` | A single `Config` dataclass holding every setting: strategy, rounds, local steps, learning rate, partition, CFL thresholds (`eps_1`, `eps_2`, `gamma_max`), straggler settings, and output paths. Defaults follow FLamby where FLamby defines them. |
| `fl_devices.py` | The core CFL machinery: `Client` (local SGD and the weight update ΔW), `Server` (cluster-wise averaging and update norms), pairwise cosine similarities between updates (Eq. 51), the optimal bipartition and its complete-linkage approximation, and the γ bound from Algorithm 5. |
| `run_experiment.py` | The main entry point. Runs one experiment in one of three modes: `fedavg` (no clustering), `cfl` (clustering with splits), or `partition` (a fixed, hand-given partition). Writes per-round metrics, similarities, split events, and final models to `results_heart8/<run_name>/`. |
| `stragglers.py` | Simulates slow clients. Each client gets a speed; in `partial` mode it runs fewer local steps, and in `drop` mode it is removed from the round if it misses the deadline. Nothing is actually waited for. |
| `baselines.py` | Non-federated reference models: one pooled model on all training data (upper bound), one model per hospital, and one model per sub-client (lower bound). Also builds the 4×4 cross-hospital accuracy matrix. |
| `calibrate.py` | Reads `eps_1`, `eps_2`, and `gamma_max` from a finished FedAvg run using one fixed rule (the round where Dice/accuracy gain flattens, then the update norms at that round). This avoids hand-picking the thresholds. |
| `run_all.sh` | The whole study in order: baselines → FedAvg → calibration → oracle partition → CFL → CFL with stragglers. Thresholds are taken from the calibration step unless `EPS1`/`EPS2` are set in the environment. |

## Pipeline

```
download.py ──► data_utils.py ──► run_experiment.py (fedavg8) ──► calibrate.py
                    │                                              │
                    ├──► baselines.py                              ▼
                    │                              run_experiment.py (oracle_centers, cfl8, cfl8_strag)
                    └── models.py, fl_devices.py, stragglers.py, config.py  (used by the above)
```

## Outputs

Each run in `results_heart8/<run_name>/` contains:

- `log.txt`: the console log
- `config.json`: the exact settings used
- `setup.json`: client names, hospital of each client, sample counts, and straggler speeds
- `metrics.csv`: one row per round (accuracy, number of clusters, ARI against the hospitals, update norms, etc.)
- `rounds.jsonl`, `cross_eval.jsonl`: per-round detail and cross-hospital accuracy
- `final.json`: final clusters, split events, and per-client results
- `similarities.npz`: the pairwise similarity matrix for every round
- `models/`: the final weights of each client (if `--no-save-models` was not given)
