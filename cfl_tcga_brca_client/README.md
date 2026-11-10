# cfl_tcga_brca_client

Clustered Federated Learning (CFL) on the Fed-TCGA-BRCA survival benchmark from FLamby, with 18 sub-clients drawn from 6 regions. The model is a 40-parameter linear Cox model, and the whole study runs in about 10 minutes on CPU or GPU.

## Quick start

```bash
bash cfl_tcga_brca_client/run_all.sh          # the full study
```

The data is already in the repo under `brca/` (`brca.csv`, `train_test_split.csv`, `centers.csv`), so there is no download step. Results are written to `results_brca18/` in the project root.

## Files

| File | What it does |
|---|---|
| `__init__.py` | Empty. Marks the folder as a Python package so the `python -m cfl_tcga_brca_client.<module>` commands work. |
| `data_utils.py` | Loads `brca/brca.csv` and merges it with `train_test_split.csv` (1088 of 1096 patients are in the split). Each patient has 39 features, an event flag `E` and a follow-up time `T`. Splits each region's training set into `clients_per_center` sub-clients (3 by default, so 18 in total), stratified by the event flag so every sub-client gets some deaths. Warns when a sub-client has fewer than 2 events. |
| `models.py` | The FLamby baseline: `Linear(39, 1)` with no activation, which outputs a Cox risk score. Also the Cox partial-likelihood loss, and Harrell's concordance index (c-index) as the metric, re-implemented so the project does not depend on `lifelines`. |
| `config.py` | A single `Config` dataclass holding every setting: strategy, rounds, local steps, learning rate, partition, CFL thresholds (`eps_1`, `eps_2`, `gamma_max`), straggler settings, and output paths. Defaults follow FLamby where FLamby defines them. |
| `fl_devices.py` | The core CFL machinery: `Client` (local SGD and the weight update ΔW, skipping batches with no events), `Server` (cluster-wise averaging and update norms), pairwise cosine similarities between updates (Eq. 51), the optimal bipartition and its complete-linkage approximation, and the γ bound from Algorithm 5. |
| `run_experiment.py` | The main entry point. Runs one experiment in one of three modes: `fedavg` (no clustering), `cfl` (clustering with splits), or `partition` (a fixed, hand-given partition). Writes per-round metrics, similarities, split events, and final models to `results_brca18/<run_name>/`. |
| `stragglers.py` | Simulates slow clients. Each client gets a speed; in `partial` mode it runs fewer local steps, and in `drop` mode it is removed from the round if it misses the deadline. Nothing is actually waited for. |
| `baselines.py` | Non-federated reference models: one pooled model on all training data (upper bound), one model per region, and one model per sub-client (lower bound). Also builds the 6×6 cross-region c-index matrix. |
| `calibrate.py` | Reads `eps_1`, `eps_2`, and `gamma_max` from a finished FedAvg run using one fixed rule (the round where c-index gain flattens, then the update norms at that round). This avoids hand-picking the thresholds. |
| `run_all.sh` | The whole study in order: baselines → FedAvg → calibration → oracle partition → CFL → CFL with stragglers. Thresholds are taken from the calibration step unless `EPS1`/`EPS2` are set in the environment. |

## Pipeline

```
brca/ ──► data_utils.py ──► run_experiment.py (fedavg18) ──► calibrate.py
              │                                                  │
              ├──► baselines.py                                  ▼
              │                          run_experiment.py (oracle_centers, cfl18, cfl18_strag)
              └── models.py, fl_devices.py, stragglers.py, config.py  (used by the above)
```

## Outputs

Each run in `results_brca18/<run_name>/` contains:

- `log.txt`: the console log
- `config.json`: the exact settings used
- `setup.json`: client names, region of each client, sample and event counts, and straggler speeds
- `metrics.csv`: one row per round (c-index, number of clusters, ARI against the regions, update norms, etc.)
- `rounds.jsonl`, `cross_eval.jsonl`: per-round detail and cross-region c-index
- `final.json`: final clusters, split events, and per-client results
- `similarities.npz`: the pairwise similarity matrix for every round
- `models/`: the final weights of each client (if `--no-save-models` was not given)
