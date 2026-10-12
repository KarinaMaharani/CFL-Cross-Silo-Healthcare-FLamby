# CFL on Fed-ISIC2019 with 18 clients (6 hospitals × 3 sub-clients)

Clustered Federated Learning — Sattler, Müller & Samek (arXiv:1910.01991), **Algorithm 5**
— on FLamby's **Fed-ISIC2019** skin-lesion classification, with each hospital split into
3 clients, FLamby's **EfficientNet-B0**, and the same **straggler simulator** as
`../cfl_ixi_client`.

Same design as the IXI study, so the two are directly comparable: the hospital of a
client is a **known cluster label**, which makes ARI, purity and the true separation gap
measurable — impossible with FLamby's 6 natural clients.

---

## 1. Files

| file | what it does |
|---|---|
| `config.py` | Every knob in one `@dataclass`. FLamby's values are the defaults (batch 64, SGD lr 0.01, 100 local updates, Adam 5e-4 ×20 epochs for the baselines). |
| `data_utils.py` | FLamby's split and transforms, the uint8 image cache, and **the 18-client partition** (§2). Train augmentations run batched on the GPU. |
| `models.py` | FLamby's **EfficientNet-B0** baseline: `efficientnet_pytorch` ImageNet weights with the classifier replaced by `Linear(1280, 8)` — 4,017,796 parameters (§4). |
| `metrics.py` | FLamby's **weighted focal loss** (γ=2, their α weights) and **balanced accuracy**, plus per-class recall and the confusion matrix. |
| `stragglers.py` | Per-client speeds, a round deadline, `partial` / `drop` behaviour, simulated round time (§5). Identical to the IXI version. |
| `fl_devices.py` | `Client`, `Server` and the CFL primitives of Algorithm 5 (Eq. 32/33 norms, Eq. 51 similarities, Eq. 48 gap, Eq. 25 bipartition, the line-18 γ test). |
| `run_experiment.py` | The round loop and all logging. Strategies: `cfl`, `fedavg`, `partition`. |
| `baselines.py` | Pooled, per-hospital and per-sub-client models, each tested on all 6 centres. |
| `calibrate.py` | Derives ε₁, ε₂, γ_max from a finished FedAvg run with one stated rule. |
| `run_all.sh` | The whole study in dependency order, for tmux. |
| `sync.sh` | rsync push/pull between the laptop and the GPU server. |

Outputs go to `../results_isic18/<run_name>/`: `config.json`, `setup.json` (client →
hospital map, sizes, **the image indices of every client**), `log.txt`, `metrics.csv`,
`rounds.jsonl`, `similarities.npz`, `cross_eval.jsonl`, `final.json`.

---

## 2. How the 18 clients are made

The recipe of `../cfl-replicate-CIFAR/cfl_cifar/data_utils.py::partition` — **one seeded
shuffle, then contiguous blocks** — applied **once per hospital**, so a sub-client only
ever holds images of its own hospital:

| hospital | train images | sub-client sizes | client ids |
|---|---|---|---|
| BCN | 9,930 | 3310 / 3310 / 3310 | 0, 1, 2 |
| HAM_vidir_molemax | 3,163 | 1055 / 1054 / 1054 | 3, 4, 5 |
| HAM_vidir_modern | 2,691 | 897 / 897 / 897 | 6, 7, 8 |
| HAM_rosendahl | 1,807 | 603 / 602 / 602 | 9, 10, 11 |
| MSK | 655 | 219 / 218 / 218 | 12, 13, 14 |
| HAM_vienna_dias | 351 | 117 / 117 / 117 | 15, 16, 17 |

Client `i` belongs to hospital `i // 3`, and that hospital is the **ground-truth
cluster**. Verified: pairwise disjoint, no client mixes hospitals, per-hospital totals
match FLamby's data sheet exactly, and the split is reproducible from `--seed` alone
(`setup.json` stores the actual image indices).

Switches: `--partition-mode equal_drop` for CIFAR's exact "drop the remainder"
arithmetic; `--partition-seed-mode global` for a single CIFAR-style RNG stream across
hospitals; `--split-test-set` to partition the test images too (by default sub-clients
share their hospital's test set).

**Planted conflict.** `--swap-centers 3,4,5` makes those hospitals swap MEL↔NV in train
*and* test, giving a second, known 2-cluster ground truth — the CIFAR label-permutation
trick on real data. `ari_vs_truth` then scores against the planted groups and
`ari_vs_centers` against the hospitals.

---

## 3. The federated protocol (FLamby's)

| | value | source |
|---|---|---|
| local work | 100 SGD steps/round | FLamby benchmark `num_updates` |
| batch size | 64 | `fed_isic2019/common.py` |
| optimiser | SGD lr 0.01 | `flamby/config_isic2019.json`, FedAvg |
| pooled/local baselines | Adam lr 5e-4, 20 epochs | `common.py` |
| loss / metric | weighted focal loss / balanced accuracy | `loss.py`, `metric.py` |
| aggregation | weighted by client size, parameters only | FLamby strategies |
| gradient clipping at 5 | **not FLamby** | without it single clients diverge: the focal α weights reach 101× |

**On `--local-updates`.** With 18 clients, 100 updates each is 3× the total work of
FLamby's 6-client setup. `run_all.sh` therefore defaults to `UPDATES=33`, which keeps
the work per hospital equal and makes these runs comparable with `../cfl_isic` (the
6-client study). Use `UPDATES=100` for protocol fidelity instead; it costs 3× the time.

---

## 4. The model

FLamby does not implement EfficientNet itself — `fed_isic2019/model.py` imports the
PyTorch port by Luke Melas-Kyriazi and swaps the head:

```python
self.base_model = EfficientNet.from_pretrained("efficientnet-b0")
nftrs = self.base_model._fc.in_features        # 1280
self.base_model._fc = nn.Linear(nftrs, 8)      # 8 ISIC classes
```

`models.py` does exactly that, so the network is the one FLamby trains:
**4,017,796 parameters**.

The architecture is **EfficientNet-B0** (Tan & Le, arXiv:1905.11946): a stem
convolution, 16 inverted-residual **MBConv** blocks with squeeze-and-excitation
(kernels 3 and 5, expansion 1 or 6, 5 resolution stages), a 1×1 head to 1280 channels,
global average pooling and a linear classifier. B0 is the base network that the paper's
compound scaling (depth × width × resolution) grows into B1–B7. The official code is
TensorFlow (`tensorflow/tpu/models/official/efficientnet`); `efficientnet_pytorch` is a
port of those same weights, which is why `pretrained=True` matters — ISIC has 351 images
at its smallest hospital, far too few to train 4 M parameters from scratch.

`--not-pretrained` runs it from random init if you want that ablation.

---

## 5. Stragglers

Identical to the IXI study. Each client has a relative speed; with a round budget of
`deadline × local_updates` nominal step-times it completes
`floor(deadline × local_updates × speed)` steps.

* `--straggler-mode partial` → fewer local steps; `drop` → misses the round entirely
  (but still receives its cluster's aggregate); `none` (default) → everyone equal.
* `--straggler-profile tiers` (default) cycles `1.0, 0.5, 0.25` over the sub-clients
  **inside each hospital**, so every hospital owns a fast, a medium and a slow client
  and straggling is orthogonal to the true clustering.
* `lognormal` / `uniform` / `manual` profiles, `--straggler-dynamic` to resample each
  round, `--straggler-seed` independent of the data seed.

**Why it matters:** fewer steps ⇒ smaller ‖Δθᵢ‖, and ‖Δθᵢ‖ is exactly what Eq. 33 tests.
In the IXI study this made CFL cluster by *compute speed* and shatter every hospital
into singletons (ARI 1.00 → 0.00). The same experiment here says whether that failure
reproduces on a second dataset.

---

## 6. The splitting criteria

A cluster splits only when all of Algorithm 5's conditions hold:

1. **Eq. 32** `‖Σ (|Dᵢ|/|D_c|) Δθᵢ‖ < ε₁` (size-weighted, as in the paper)
2. **Eq. 33** `maxᵢ ‖Δθᵢ‖ > ε₂`
3. **Eq. 25** the cut with the lowest maximum cross-cluster similarity — solved
   **exactly** for clusters of ≤16 clients (32,767 cuts) and by complete linkage above
   that; `linkage_agrees` and `exact_bipartition` are logged every round
4. **Line 18** `γ_max < √((1 − α_cross,max)/2)`, with γ_max measured by the probe
   (`--gamma-auto --gamma-probe-every 5`): two i.i.d. halves of one client trained from
   the same weights, so their update dissimilarity is pure noise and
   `γ̂ = √((1 − α_halves)/2)`

Guards that are **not** in the paper (`min_split_round`, `split_cooldown`,
`min_cluster_size`) are all off by default. ε₁/ε₂ come from `calibrate.py`, which uses
the first round whose balanced-accuracy gain over 10 rounds is below 0.005.

---

## 7. Running it

Requirements: `torch` (CUDA), `efficientnet_pytorch`, `numpy`, `scipy`, `scikit-learn`,
`pandas`, `pillow`. Data: `../isic/ISIC_2019_Training_Input_preprocessed/` (23,247
preprocessed jpgs) and `../isic/train_test_split.csv`. The first run builds
`../isic/cache_220.npy` (3.2 GB, ~5 min); later runs memory-map it.

```bash
cd "<...>/cfl-replicate-flamby"          # the project root, NOT this folder

# smoke test (~10 min, mostly the cold cache read)
python -m cfl_isic_client.run_experiment --run-name smoke --strategy cfl --rounds 2 \
    --local-updates 3 --split-schedule 2 --straggler-mode partial

# the whole study (see the cost note below)
bash cfl_isic_client/run_all.sh
```

**Cost on an RTX 3050** (measured: 0.23 s per training step at batch 64, and ~80 s per
round for evaluation unless `--eval-every` is raised — the default is 5):

| | per round | 40 rounds |
|---|---|---|
| `--local-updates 33` (compute-matched, the `run_all.sh` default) | ~2.5 min | **~1.7 h** |
| `--local-updates 100` (FLamby's number) | ~7 min | **~4.7 h** |

The full 5-job study is therefore roughly **7 h** at 33 updates, or **19 h** at 100.
On an A100 expect 2–3× faster, and ~4 GB of VRAM per run, so 3 runs fit side by side.
For the server workflow (rsync + tmux) see `sync.sh` and §8 of
`../cfl_ixi_client/README.md`.

---

## 8. Correspondence with the plain CFL reference

Identical to the mapping in `../cfl_ixi_client/README.md` §9 (upstream
`clustered-federated-learning/` → this code), with the dataset-specific parts swapped:

| upstream | here |
|---|---|
| `models.py::ConvNet` (2D EMNIST CNN) | `models.py::Baseline` (EfficientNet-B0) |
| `train_op` / `eval_op` (cross-entropy, accuracy) | `Client.compute_weight_update` / `metrics.evaluate` (weighted focal loss, balanced accuracy) |
| `data_utils.py::split_noniid` (Dirichlet) | `data_utils.py::partition_pool` (per-hospital CIFAR-style split) |
| `Server.select_clients(frac)` | `stragglers.py` |
| notebook round loop | `run_experiment.py` (+ the γ test, ARI/purity, cross-site evaluation, stragglers) |

---

## 9. What to compare with the IXI study

Both studies use the same partition recipe, the same straggler model and the same CFL
code, so the interesting comparison is **how the two datasets differ**:

| | Fed-IXI (9 clients) | Fed-ISIC2019 (18 clients) |
|---|---|---|
| heterogeneity | scanner only, mild | class mix + camera + size, strong |
| client sizes | 19–83 volumes | 117–3310 images |
| noise floor γ̂ | 0.02–0.10 | ~0.6 (measured in `../cfl_isic`) |
| CFL result | correct split, ARI 0.50, +0.017 Dice for the outlier | to be measured |
| with stragglers | shattered to singletons, ARI 0.00 | to be measured |

The quantities to read off: `ari` and `pure_clusters` per round, `respects_centers` on
each split event, `gap_true` vs `gap_est`, `gamma_hat` against the between-hospital
similarity, and CFL vs `oracle_centers` vs the per-client baselines.
