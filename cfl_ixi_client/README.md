# CFL on Fed-IXI with 9 clients (3 hospitals × 3 sub-clients)

Clustered Federated Learning — Sattler, Müller & Samek, *Clustered Federated Learning:
Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints*
(arXiv:1910.01991), **Algorithm 5** — on FLamby's **Fed-IXI** brain-mask segmentation,
with each hospital split into 3 clients, a 3D U-Net built from
[wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), and a **straggler
simulator** for clients that compute at different speeds.

**Why 9 clients.** Fed-IXI has only 3 natural clients, so "did CFL cluster correctly?"
has no ground-truth answer and a bipartition is almost trivial. Splitting each hospital
into 3 i.i.d. sub-clients gives a **known correct clustering** (the hospital), so the
recovered partition can be scored with **ARI**, cluster purity and the paper's
separation gap — exactly the way the CIFAR replication scores its 4 planted clusters.

---

## 1. Files

| file | what it does |
|---|---|
| `config.py` | Every knob in one `@dataclass`. FLamby's values are the defaults where FLamby defines them (batch 2, lr 1e-3, 100 local updates); CFL defaults follow `../cfl-replicate-CIFAR`. Any field is overridable from the command line. |
| `data_utils.py` | Reads IXI Tiny, applies FLamby's preprocessing (T1, area-resize to 48×60×48, z-score, truncated label), reproduces FLamby's train/test split, caches the result as `.npz`, and **builds the 9 clients** (§2). Also `TensorClientData`, which serves batches from VRAM with FLamby's `DataLoaderWithMemory` semantics. |
| `models.py` | The **3D U-Net** (§4): FLamby's Fed-IXI network (246 156 parameters), built from the pytorch-3dunet building blocks. Run `python -m cfl_ixi_client.models` to print its layers. |
| `metrics.py` | FLamby's Dice loss and soft-Dice metric, verbatim, plus a thresholded foreground Dice and per-subject scores (needed for paired confidence intervals). |
| `stragglers.py` | The straggler simulator (§5): per-client speeds, a round deadline, `partial` / `drop` behaviour, and simulated round time. |
| `fl_devices.py` | `Client`, `Server` and the CFL primitives of Algorithm 5: weight updates, cluster-wise averaging, Eq. 32/33 norms, cosine similarities (Eq. 51), the separation gap (Eq. 48), the γ bound (line 18) and the optimal bipartition (Eq. 25) solved **exactly** and by the reference complete-linkage heuristic. |
| `run_experiment.py` | The communication-round loop and all logging. Strategies: `cfl`, `fedavg` (never splits), `partition` (a fixed grouping, for oracles). |
| `baselines.py` | Non-federated reference points: pooled, per-hospital, per-sub-client, each tested on every hospital. |
| `calibrate.py` | Reads ε₁, ε₂ and γ_max off a finished FedAvg run with one fixed, stated rule (§6). |
| `run_all.sh` | The whole study in dependency order; made for `tmux` (§8). |

Outputs go to `../results_ixi9/<run_name>/`:

| file | contents |
|---|---|
| `config.json` | the exact configuration used |
| `setup.json` | client → hospital map, sizes, **the scan indices of every client** (so a split can be replayed), straggler speeds |
| `log.txt` | human-readable log, including every split and every γ rejection |
| `metrics.csv` | one row per round: Dice, ARI, purity, norms, gaps, γ, straggler steps, simulated time |
| `rounds.jsonl` | one JSON per round with the full per-client and per-cluster detail |
| `similarities.npz` | the 9×9 cosine-similarity matrix of every round |
| `cross_eval.jsonl` | every client model × every hospital test set, every *k* rounds |
| `final.json` | final clusters, ARI, split events, per-subject Dice for paired tests |
| `models/` | final client weights (`--no-save-models` to skip) |

---

## 2. How the 9 clients are made

Copied from `../cfl-replicate-CIFAR/cfl_cifar/data_utils.py::partition`, which does:

```python
per  = n_train // n_clients
idcs = rng.permutation(n_train)                                   # one seeded shuffle
client_idcs = [idcs[i*per:(i+1)*per] for i in range(n_clients)]   # contiguous blocks
```

i.e. **shuffle once with a seeded RNG, then cut the shuffled list into equal contiguous
blocks**. Random, disjoint, equal-sized, not stratified.

Here the same recipe is applied **once per hospital**, so a sub-client can only ever
hold scans of its own hospital:

| hospital | train scans | sub-client sizes | client ids |
|---|---|---|---|
| Guys | 249 | 83 / 83 / 83 | 0, 1, 2 |
| HH | 145 | 49 / 48 / 48 | 3, 4, 5 |
| IOP | 59 | 20 / 20 / 19 | 6, 7, 8 |

Client `i` belongs to hospital `i // clients_per_center`, and **that hospital is the
ground-truth cluster label** used for ARI, purity and `gap_true`.

Two switchable differences from CIFAR:

* **The remainder is kept.** `np.array_split` gives 49/48/48 instead of discarding
  scans. `--partition-mode equal_drop` reproduces CIFAR's `n // k` arithmetic exactly
  (83/83/83, 48/48/48, 19/19/19 — 3 scans unused).
* **Each hospital gets its own RNG,** seeded `[seed, center_id]`, so changing one
  hospital's client count does not reshuffle the others. `--partition-seed-mode global`
  uses a single CIFAR-style stream across hospitals.

Reproducibility: the split depends only on `--seed`, and `setup.json` stores the actual
scan indices of every client. Verified: same seed → identical split, different seed →
different split, clients pairwise disjoint, no client mixes hospitals.

Test sets: sub-clients of a hospital **share that hospital's test set** (62 / 36 / 15
scans), so per-client scores are comparable and not noisier than they need to be.
`--split-test-set` partitions the test scans too.

---

## 3. The federated protocol (FLamby's)

| | value | source |
|---|---|---|
| local work per round | 100 SGD steps | FLamby benchmark `num_updates` |
| batch size | 2 | `fed_ixi/common.py` |
| optimiser | SGD, lr 1e-3 (no momentum) | `flamby/config_ixi.json` FedAvg |
| aggregation | weighted by client size | FLamby FedAvg |
| BatchNorm statistics | stay local, never aggregated | FLamby strategies + CFL reference |
| loss / metric | Dice loss / soft Dice (+ foreground Dice) | `fed_ixi/{loss,metric}.py` |

---

## 4. The model

FLamby's Fed-IXI network, **implemented with the building blocks of the pytorch-3dunet
zip** (MIT, (c) 2018 Adrian Wolny): `create_conv`, `SingleConv`, `DoubleConv`,
`Encoder`, `Decoder`, `create_encoders/decoders`, the upsampling strategies and
`UNet3D`. The 2D variants, ResNet/SE blocks and training plumbing are dropped.

The architecture family is that of **Cicek et al., 3D U-Net (arXiv:1606.06650)** -- two
3x3x3 convolutions per level, 2x2x2 max pooling, skip connections at equal resolution,
a 1x1x1 convolution to the label channels, batch normalisation before each activation,
and the paper's rule of *doubling the channels before pooling*. FLamby keeps all of
that and picks a much smaller instance of it, because IXI Tiny volumes are 48x60x48 and
the task is binary brain extraction rather than 3-class sparse annotation.

Setting `f_maps=(16, 32, 64)` makes pytorch-3dunet's channel rule (`conv1 -> out//2`,
`conv2 -> out`) reproduce FLamby's widths exactly:

| stage | channels |
|---|---|
| encoder 1 | 1 -> 8 -> 16 (first conv has **no** BatchNorm, so it keeps its bias) |
| encoder 2 | 16 -> 16 -> 32 |
| encoder 3 (bottom) | 32 -> 32 -> 64 |
| decoder 1 | (64 up + 32 skip) = 96 -> 32 -> 32 |
| decoder 2 | (32 up + 16 skip) = 48 -> 16 -> 16 |
| classifier | 16 -> 2, kernel 1, then softmax over channels |

Two small extensions to the vendored grammar were needed to express FLamby exactly:

* **`p` = PReLU** in the layer order, so FLamby's conv -> BatchNorm -> PReLU block is
  the order `"cbp"` (pytorch-3dunet offered only ReLU / LeakyReLU / ELU);
* **`first_conv_order`**, because fepegar/unet (the package FLamby vendors) runs the
  *first* convolution of the *first* block without normalisation.

**Verification:** the result has **246,156 parameters and an identical
parameter-shape multiset** to the FLamby model used in `../cfl_ixi`, so numbers stay
comparable with the earlier 3-client runs.

| FLamby's setting | here |
|---|---|
| in_channels 1, out_classes 2 | same |
| num_encoding_blocks 3 | `num_levels=3` |
| out_channels_first_layer 8 | `f_maps=16` (block width: conv1 -> 8, conv2 -> 16) |
| normalization "batch", activation "PReLU" | `layer_order="cbp"` |
| upsampling_type "linear" | `upsample="trilinear"` |
| padding 1 | `conv_padding=1` |
| batch size 2, AdamW lr 1e-3 (pooled), Dice loss | `config.py` / `metrics.py` |

Overrides for ablations: `--f-maps`, `--num-levels`, `--layer-order`, `--upsample`
(`trilinear` / `nearest` / `deconv`). For reference only, the Cicek paper's own instance
would be `--f-maps 64 --num-levels 4 --layer-order cbr --upsample deconv` (about 18.6 M
parameters against the paper's 19,069,955; the gap is the decoder convention plus padded
vs valid convolutions). It is **not** part of this study.

Input 48x60x48 is not a power of two; pooling floors odd sizes (60 -> 30 -> 15) and each
decoder interpolates back to its skip connection's exact size, so 3 levels are fine.

## 5. Stragglers

Real cross-silo rounds are synchronous: the server waits. A slow hospital therefore
either returns a **partial** update or **misses the deadline**. Both are simulated —
no wall-clock time is actually wasted.

Each client has a relative speed `s_i` (1.0 = finishes 100 steps in exactly one round
budget). With a budget of `deadline × local_updates` nominal step-times:

```
capacity_i = floor(deadline × local_updates × s_i)

--straggler-mode partial   steps_i = clip(capacity_i, min_updates, local_updates)
--straggler-mode drop      steps_i = local_updates if capacity_i >= local_updates else DROPPED
--straggler-mode none      everyone runs the full local_updates   (default)
```

Dropped clients contribute no Δθ but **still receive their cluster's aggregate**, so
cluster members stay in sync. A cluster is only considered for splitting in rounds
where **all** its members reported, since a bipartition must place every member.

Speed profiles (`--straggler-profile`):

| profile | meaning |
|---|---|
| `tiers` (default) | cycle `--straggler-tiers 1.0,0.5,0.25` over the sub-clients **within each hospital**, so every hospital owns one fast, one medium and one slow client. Straggling is then *orthogonal* to the ground-truth clustering and cannot be mistaken for it. |
| `lognormal` | `s = exp(N(0, σ))`, clipped to [0.05, 4] |
| `uniform` | `s = 1 ± σ`, clipped to [0.05, 4] |
| `manual` | `--straggler-speeds 1.0,0.8,...` one per client |

`--straggler-dynamic` resamples speeds every round (transient load) instead of fixing
them per client (hardware heterogeneity). Everything is driven by `--straggler-seed`,
which is **independent of the data seed**, so the straggler pattern can be changed
without reshuffling the partition.

Simulated round time (= `max_i steps_i / s_i`, or the full budget if anyone was cut
off) is accumulated in `cum_round_time`, giving accuracy-vs-simulated-time curves next
to accuracy-vs-rounds.

**Why this matters for CFL:** fewer local steps ⇒ a smaller `‖Δθ_i‖`, which is exactly
the quantity Eq. 33 tests (`max_i ‖Δθ_i‖ > ε₂`). Stragglers therefore *look converged*
to the splitting criteria. `steps_*` and `norm_*` are logged per client every round so
this can be measured rather than assumed.

---

## 6. The splitting criteria, and how ε₁ / ε₂ / γ are chosen

A cluster splits only when all of Algorithm 5's conditions hold at once:

1. **Line 14, stationarity (Eq. 32):** `‖Σ_i (|D_i|/|D_c|) Δθ_i‖ < ε₁` — the size-weighted
   mean update is small (`--unweighted-mean-norm` for the plain mean).
2. **Line 14, incongruence (Eq. 33):** `max_i ‖Δθ_i‖ > ε₂`.
3. **Line 16/25:** the cut with the lowest maximum cross-cluster similarity, solved
   exactly (255 cuts for 9 clients) and compared with complete linkage each round
   (`linkage_agrees` in the logs).
4. **Line 18:** `γ_max < sqrt((1 − α_cross,max)/2)` — the split must be bigger than noise.

The paper gives no values for ε₁, ε₂ or γ_max. `calibrate.py` derives them from a
FedAvg run with one fixed rule: the first round whose Dice gain over the previous 10
rounds is below 0.005 defines "stationary"; ε₁ is the weighted mean norm there, ε₂ sits
just below the smallest `max_i ‖Δθ_i‖` seen in the run.

γ_max is **measured, not guessed**: `--gamma-probe-every k` trains two disjoint i.i.d.
halves of one client from the same weights every *k* rounds. They share a distribution,
so their update dissimilarity is pure noise, and the paper's own intra-cluster bound
gives `γ̂ = sqrt((1 − α_halves)/2)`. `--gamma-auto` feeds the median of the last 3 probes
into condition 4. (This estimator is an addition of ours, not from the paper — say so
in the thesis.)

Guards that are **not** in the paper are all off by default: `--min-split-round 0`,
`--split-cooldown 0`, `--min-cluster-size 1`.

---

## 7. Running it

Requirements: `torch` (CUDA build), `numpy`, `scipy`, `scikit-learn`, `pandas`,
`nibabel`. Data must be at `../ixi/7kd5wj7v7p-3/IXI_sample` (the Mendeley IXI Tiny
release). The first run preprocesses all 566 volumes once into `../cache/` (~1–2 min,
0.4 GB); later runs load it in seconds.

```bash
cd "<...>/cfl-replicate-flamby"          # the project root, NOT this folder

# 30-second sanity check (tiny rounds, forced splits, stragglers on)
python -m cfl_ixi_client.run_experiment --run-name smoke --strategy cfl --rounds 3 \
    --local-updates 4 --split-schedule 2,3 --straggler-mode partial --no-save-models

# 1. reference points
python -m cfl_ixi_client.baselines

# 2. FedAvg: the no-split reference AND the calibration run
python -m cfl_ixi_client.run_experiment --run-name fedavg9 --strategy fedavg \
    --rounds 60 --gamma-probe-every 5
python -m cfl_ixi_client.calibrate --run fedavg9          # prints --eps-1 / --eps-2

# 3. oracle: the ground-truth grouping, trained from round 1
python -m cfl_ixi_client.run_experiment --run-name oracle_centers --strategy partition \
    --partition "0,1,2|3,4,5|6,7,8" --rounds 60

# 4. CFL, paper criteria only
python -m cfl_ixi_client.run_experiment --run-name cfl9 --strategy cfl --rounds 60 \
    --eps-1 <e1> --eps-2 <e2> --gamma-auto --gamma-probe-every 5

# 5. CFL under stragglers
python -m cfl_ixi_client.run_experiment --run-name cfl9_strag --strategy cfl --rounds 60 \
    --eps-1 <e1> --eps-2 <e2> --gamma-auto --gamma-probe-every 5 \
    --straggler-mode partial --straggler-profile tiers --straggler-tiers 1.0,0.5,0.25
```

Or the whole thing: `EPS1=... EPS2=... bash cfl_ixi_client/run_all.sh`.

Measured cost. One training step of this U-Net (batch 2, 48×60×48) takes **55.7 ms** on
an RTX 3050 laptop GPU, and evaluation 6 ms per volume. A round is 9 clients × 100 steps
plus 339 volume-evaluations, i.e. **~52 s/round**, so a 60-round run is **~50 min** and
the whole `run_all.sh` roughly **4 h**. On an A100 expect **15–25 s/round** (~15–25 min
per run, ~1–1.5 h for the full study): the model is small and the batch is 2, so the GPU
is launch-latency bound rather than compute bound and you will not see the full A100
speed-up. Each run needs only ~2 GB of VRAM, so on a 40/80 GB A100 you can run 3–4 of
them side by side and finish the study in ~30–40 min of wall clock.

---

## 8. Running it on a WSL2 server with tmux

**Why tmux:** it keeps the run alive when your SSH session or laptop drops, and lets you
reattach later from anywhere.

### One-time setup inside WSL2

```bash
# in Windows PowerShell: install WSL2 + Ubuntu, then reboot
wsl --install -d Ubuntu

# inside Ubuntu
sudo apt update && sudo apt install -y tmux python3-venv python3-pip
nvidia-smi                 # must already work: WSL2 uses the *Windows* NVIDIA driver,
                           # do NOT install a Linux NVIDIA driver inside WSL

python3 -m venv ~/venvs/cfl && source ~/venvs/cfl/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install numpy scipy scikit-learn pandas nibabel
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**The project files.** Windows drives are mounted under `/mnt/c`, so you can run in
place:

```bash
cd "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/...revamp-2/cfl-replicate-flamby"
```

That works, but `/mnt/c` I/O is slow. For long runs, copy the project onto the Linux
filesystem instead (several times faster when building the cache):

```bash
mkdir -p ~/skripsi && cp -r "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/...revamp-2/cfl-replicate-flamby" ~/skripsi/
cd ~/skripsi/cfl-replicate-flamby
```

Copy results back with `cp -r results_ixi9 "/mnt/c/.../cfl-replicate-flamby/"` when done.

### Running under tmux

```bash
tmux new -s cfl                 # create a session called "cfl"
source ~/venvs/cfl/bin/activate
cd ~/skripsi/cfl-replicate-flamby
bash cfl_ixi_client/run_all.sh  # or any single python -m ... command

# detach and leave it running:   Ctrl-b  then  d
exit                            # you can even close the terminal / disconnect SSH
```

Reattach later, from a new terminal or after a reboot of your laptop (not the server):

```bash
wsl                             # if you left WSL entirely
tmux ls                         # list sessions:  cfl: 1 windows (created ...)
tmux attach -t cfl              # back to the live output
```

Useful inside tmux:

| keys / command | effect |
|---|---|
| `Ctrl-b` `d` | detach, leave everything running |
| `Ctrl-b` `c` | new window (e.g. to run `watch -n5 nvidia-smi`) |
| `Ctrl-b` `n` / `p` | next / previous window |
| `Ctrl-b` `[` | scroll mode (arrows / PgUp; `q` to exit) |
| `tmux kill-session -t cfl` | stop everything in that session |

Monitoring from a second window (or a second SSH login):

```bash
tail -f results_ixi9/cfl9/log.txt                 # live training log
grep -E "SPLIT|REJECTED" results_ixi9/cfl9/log.txt
watch -n 5 nvidia-smi                             # GPU use
```

**Two WSL2 gotchas.** (1) WSL shuts down when the last session closes — a tmux session
inside WSL survives SSH drops but *not* `wsl --shutdown` or a Windows reboot; for
overnight runs disable Windows sleep. (2) Memory: add `~/.wslconfig` on the Windows side
(`[wsl2]` `memory=12GB`) if the preprocessing step gets OOM-killed.

If you prefer no tmux: `nohup bash cfl_ixi_client/run_all.sh > run.out 2>&1 &` then
`tail -f run.out`.

---

## 9. Where this code corresponds to the plain CFL reference

`../clustered-federated-learning/` is the authors' reference implementation (felisat),
written around an EMNIST demo notebook. Everything here is a re-implementation of it for
Fed-IXI; this table says which piece became what.

| upstream (`clustered-federated-learning/`) | here | what changed |
|---|---|---|
| `fl_devices.py::copy(target, source)` | `fl_devices.py::copy_` | identical (renamed so it does not shadow the `copy` module) |
| `fl_devices.py::subtract_` | `fl_devices.py::subtract_` | identical |
| `fl_devices.py::reduce_add_average(targets, sources)` | `reduce_add_average(targets, sources, weights=None)` | optional size weighting (FLamby's FedAvg weights clients by \|D_i\|); `targets` may be larger than `sources`, so a dropped straggler still receives its cluster's aggregate |
| `fl_devices.py::flatten` | `fl_devices.py::flatten` | identical |
| `fl_devices.py::pairwise_angles` (Eq. 51) | `fl_devices.py::pairwise_angles` | float64 on CPU; rows of clients that did not report are left NaN |
| `Server.cluster_clients` (complete linkage on -S) | `complete_linkage_bipartition` **and** `optimal_bipartition` | upstream's greedy heuristic is kept, and the exact arg min of Eq. 25 is computed too; `linkage_agrees` logs when they differ |
| `Server.compute_max_update_norm` (Eq. 33) | same name | taken over the clients that reported this round |
| `Server.compute_mean_update_norm` (Eq. 32) | `compute_mean_update_norm(..., weighted=True)` | the paper's size-weighted form; `--unweighted-mean-norm` restores upstream's plain mean |
| `Server.aggregate_clusterwise` | same name, extra args | adds BatchNorm-buffer policy and straggler participation |
| `Server.cache_model` | -- | replaced by `models/` checkpoints and `cross_eval.jsonl` |
| `Server.select_clients(frac)` | `stragglers.py` | random client sampling replaced by an explicit speed/deadline model (`partial` / `drop`) |
| `Client.compute_weight_update(epochs)` | `Client.compute_weight_update(num_updates)` | FLamby counts *steps*, not epochs, and the straggler schedule sets the count each round |
| `Client.reset`, `Client.synchronize_with_server` | same names | `synchronize_with_server` also copies BatchNorm buffers |
| `train_op` / `eval_op` | `Client.compute_weight_update` / `metrics.evaluate` | Dice loss instead of cross-entropy, Dice metrics instead of accuracy |
| `FederatedTrainingDevice` base class | -- | folded into `Client` / `Server`; no shared behaviour was left |
| `data_utils.py::split_noniid` (Dirichlet label skew) | `data_utils.py::partition_pool` | label-skew simulation is not needed: clients are real hospitals, split CIFAR-style inside each hospital (section 2) |
| `data_utils.py::CustomSubset` | `data_utils.py::TensorClientData` | volumes are held in VRAM; no torchvision transforms |
| `models.py::ConvNet` (2D, EMNIST) | `models.py::UNet3D` | the 3D U-Net of section 4 |
| `helper.py::ExperimentLogger`, `display_train_stats` | logging in `run_experiment.py` (`metrics.csv`, `rounds.jsonl`, `log.txt`) | no live notebook plots: runs are head-less over SSH/tmux, so everything is written to disk and plotted afterwards |
| `clustered_federated_learning.ipynb` (the round loop) | `run_experiment.py` | adds Algorithm 5 line 18 (the gamma test the notebook omits), FedAvg and fixed-partition strategies, ARI/purity against the known hospitals, cross-site evaluation, stragglers, and per-round logs |

The CFL mathematics is unchanged from upstream: the Eq. 32/33 criteria, Eq. 51 cosine
similarities, the Eq. 25 bipartition and the Eq. 48 separation gap. What is added is
Algorithm 5 line 18, the measured noise floor that supplies its gamma_max, and the
logging needed to score a split against a known ground truth.

## 10. What to look at in the results

* **ARI and `pure_clusters` per round** (`metrics.csv`) — did CFL recover the hospitals?
  ARI = 1.0 means the 9 clients were grouped exactly into Guys / HH / IOP.
* **`respects_centers` on each split event** (`final.json`) — a split that cuts *through*
  a hospital is a false split; the log prints `CUTS THROUGH A CENTRE` when it happens.
* **`gap_true` vs `gap_est`** — `gap_true` uses the hospital labels (Eq. 48 with the
  ground truth), `gap_est` the cut CFL actually proposes. Corollary 1 says a positive
  gap guarantees a correct split.
* **`alpha_halves` / `gamma_hat`** — the measured noise floor. Compare with the
  similarity *between* hospitals: if they are the same size, the signal is noise.
* **`steps_*`, `n_dropped`, `cum_round_time`** — the straggler accounting.
* **CFL vs `oracle_centers` vs `baselines`** — the oracle is what a perfect clustering
  achieves; per-client baselines are what full separation costs.
