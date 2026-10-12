"""Regenerate analysis.ipynb from this single source of truth.

    python tools/build_notebook.py            # write analysis.ipynb
    python tools/build_notebook.py --execute  # write it and run it (stores outputs)
"""
import argparse
import os
import subprocess
import sys

import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Clustered Federated Learning on FLamby **Fed-IXI**

Sattler, Müller & Samek, *Clustered Federated Learning* (arXiv:1910.01991) — Algorithm 5 —
applied to the Fed-IXI brain-mask segmentation task of FLamby
(Ogier du Terrail et al., NeurIPS 2022 Datasets & Benchmarks).

| | CIFAR replication | this study |
|---|---|---|
| clients | 20 synthetic | **3 real hospitals** (Guys, HH, IOP) — natural split |
| ground truth clusters | 4 label permutations | **unknown** (or the center, for virtual sub-clients) |
| model | 5-layer CNN | FLamby 3D U-Net (246 k params) |
| loss / metric | CE / accuracy | Dice loss / Dice (FLamby soft Dice + hard foreground Dice) |
| local training | 3 epochs | 100 SGD updates, lr 1e-3, batch 2 (FLamby FedAvg) |

Because there is no planted cluster structure, *"did CFL split correctly?"* has no free answer.
This notebook builds the evidence in layers:

1. **Data** — how different are the three centers at the input level?
2. **Baselines** — Pooled / Local / FedAvg, and the 3×3 cross-site matrix.
3. **Update geometry** — cosine similarities & norms under FedAvg (ε calibration, which bipartition CFL would propose, and how stable that proposal is).
4. **CFL run** — when it splits, what it gains.
5. **Oracle** — every one of the 5 possible partitions of 3 clients trained to convergence: where does CFL's partition rank, per client, with paired bootstrap CIs.
6. **Virtual sub-clients** — each center cut into k clients, so the center *is* a ground truth: ARI and separation gap, like the CIFAR experiment.
""")

code(r"""
import json, os, glob, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

RESULTS = "results"
CENTERS = ["Guys", "HH", "IOP"]
RUN_FEDAVG_FLAMBY = os.environ.get("RUN_FEDAVG_FLAMBY", "fedavg_flamby")
RUN_FEDAVG = os.environ.get("RUN_FEDAVG", "fedavg_long")
RUN_CFL    = os.environ.get("RUN_CFL", "cfl")
RUN_VIRTUAL = os.environ.get("RUN_VIRTUAL", "cfl_virtual3")
PARTITION_RUNS = {                       # partition label -> run name
    "{Guys,HH,IOP}":    RUN_FEDAVG,
    "{Guys,HH}{IOP}":   "part_01_2",
    "{Guys,IOP}{HH}":   "part_02_1",
    "{HH,IOP}{Guys}":   "part_12_0",
    "{Guys}{HH}{IOP}":  "part_0_1_2",
}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

def exists(name): return name and os.path.exists(os.path.join(RESULTS, name, "metrics.csv"))

def load_run(name):
    if not exists(name): return None
    d = os.path.join(RESULTS, name)
    r = {"name": name, "dir": d,
         "cfg": json.load(open(os.path.join(d, "config.json"))),
         "setup": json.load(open(os.path.join(d, "setup.json"))),
         "metrics": pd.read_csv(os.path.join(d, "metrics.csv")),
         "rounds": [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]}
    for fn, key in [("final.json", "final")]:
        p = os.path.join(d, fn); r[key] = json.load(open(p)) if os.path.exists(p) else None
    p = os.path.join(d, "cross_eval.jsonl")
    r["cross"] = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    p = os.path.join(d, "similarities.npz"); r["S"] = np.load(p)["S"] if os.path.exists(p) else None
    return r

print("runs:", sorted(os.path.basename(p) for p in glob.glob(RESULTS + "/*") if os.path.isdir(p)))
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Data-level heterogeneity

FLamby's Fed-IXI clients are defined by acquisition site. The three London sites used
different scanners (Guy's: Philips 1.5 T, HH: Philips 3 T, IOP: GE 1.5 T), so we expect
**feature-distribution skew** (φ(x) differs) rather than the **concept shift** (φ(y|x)
differs) that CFL was designed for — a brain mask is a brain mask everywhere. That
distinction is the key to interpreting everything below: CFL's theory guarantees a
useful split only for *incongruent* clients.

Statistics are computed on the raw 83×44×55 volumes (before FLamby's per-volume z-score,
which removes most of the global intensity offset).
""")

code(r"""
meta = pd.read_csv("cache/subjects.csv")
parts = pd.read_csv("ixi/7kd5wj7v7p-3/IXI_sample/participants.csv")
meta = meta.merge(parts[["FOLDER_NAME", "AGE", "SEX_ID (1=m, 2=f)"]].rename(
    columns={"FOLDER_NAME": "name", "SEX_ID (1=m, 2=f)": "sex"}), on="name", how="left")
display(pd.crosstab(meta.center, meta.split, margins=True))
cols = ["raw_max", "raw_p99", "raw_brain_mean", "raw_brain_std", "raw_bg_mean",
        "raw_brain_fraction", "brain_fraction_48", "AGE"]
display(meta.groupby("center")[cols].agg(["mean", "std"]).round(3).T)

fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
for a, c in zip(ax, ["raw_brain_mean", "raw_p99", "brain_fraction_48", "AGE"]):
    a.boxplot([meta.loc[meta.center == k, c].dropna() for k in CENTERS], labels=CENTERS)
    a.set_title(c)
plt.tight_layout(); plt.show()
""")

code(r"""
# mean z-scored image and mean label per center (middle slices) + center difference maps
z = np.load("cache/ixi_tiny_T1_48x60x48.npz")
X, Y = z["x"][:, 0], z["y"]
cid = meta.center_id.values
means = [X[cid == c].mean(0) for c in range(3)]
labs = [Y[cid == c].mean(0) for c in range(3)]
fig, ax = plt.subplots(3, 5, figsize=(14, 8))
for c in range(3):
    ax[c, 0].imshow(means[c][24].T, cmap="gray", origin="lower"); ax[c, 0].set_ylabel(CENTERS[c])
    ax[c, 1].imshow(means[c][:, 30].T, cmap="gray", origin="lower")
    ax[c, 2].imshow(labs[c][24].T, cmap="magma", origin="lower", vmin=0, vmax=1)
    d = means[c] - np.mean(means, 0)
    ax[c, 3].imshow(d[24].T, cmap="coolwarm", origin="lower", vmin=-1, vmax=1)
    ax[c, 4].hist(X[cid == c][Y[cid == c] == 1][::50], bins=80, alpha=.7, density=True, label="brain")
    ax[c, 4].hist(X[cid == c][Y[cid == c] == 0][::50], bins=80, alpha=.5, density=True, label="bg")
for a, t in zip(ax[0], ["mean image (sag.)", "mean image (cor.)", "mean label", "mean - global mean", "z-scored intensities"]):
    a.set_title(t)
ax[0, 4].legend(); [a.grid(False) for a in ax[:, :4].ravel()]
plt.tight_layout(); plt.show()

# a simple, model-free distance between centers: intensity histogram distance (brain voxels)
bins = np.linspace(-2, 5, 141)
H = [np.histogram(X[cid == c][Y[cid == c] == 1], bins=bins, density=True)[0] for c in range(3)]
H = [h / h.sum() for h in H]
tv = np.array([[0.5 * np.abs(H[i] - H[j]).sum() for j in range(3)] for i in range(3)])
display(pd.DataFrame(tv, CENTERS, CENTERS).round(3).style.set_caption(
    "total-variation distance of brain-intensity histograms (z-scored)"))
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Baselines — Pooled, Local, FedAvg (FLamby protocol)

* **Pooled / Local i**: AdamW, lr 1e-3, batch 2, 10 epochs (`cfl_ixi/baselines.py`).
* **FedAvg (FLamby)**: SGD lr 1e-3, 100 local updates, `get_nb_max_rounds(100)` = 7 rounds.
* **FedAvg (long)**: same but 60 rounds — the non-splitting reference for CFL.

The **cross-site matrix** (row = training data, column = test center) is the most direct
evidence about partition quality: if the model trained on center *i* is as good on center *j*
as *j*'s own model, there is nothing to gain by separating them.
""")

code(r"""
B = {}
for k in ["pooled", "local0", "local1", "local2"]:
    p = os.path.join(RESULTS, "baselines", k + ".json")
    if os.path.exists(p): B[k] = json.load(open(p))
rows = {"pooled": "Pooled", "local0": "Local Guys", "local1": "Local HH", "local2": "Local IOP"}
for metric in ["dice", "dice_fg"]:
    M = pd.DataFrame({rows[k]: v[metric] for k, v in B.items()}, index=CENTERS).T
    fl = {}
    for label, name in [("FedAvg (FLamby, 7 rounds)", RUN_FEDAVG_FLAMBY), ("FedAvg (60 rounds)", RUN_FEDAVG)]:
        r = load_run(name)
        if r and r["final"]:
            fl[label] = [r["final"]["cross"][metric][c][c] for c in range(3)]
    M = pd.concat([M, pd.DataFrame(fl, index=CENTERS).T])
    w = np.array([62, 36, 15]); M["mean"] = M[CENTERS].mean(1); M["pooled-weighted"] = (M[CENTERS] * w).sum(1) / w.sum()
    display(M.round(4).style.background_gradient(axis=None, subset=CENTERS, cmap="viridis")
            .set_caption(metric + "  (rows: model, cols: test center)"))
""")

code(r"""
if B:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, metric in zip(ax, ["dice", "dice_fg"]):
        M = np.array([B[k][metric] for k in ["local0", "local1", "local2"] if k in B])
        im = a.imshow(M, cmap="viridis"); plt.colorbar(im, ax=a)
        for i in range(M.shape[0]):
            for j in range(3): a.text(j, i, "{:.4f}".format(M[i, j]), ha="center", va="center", color="w", fontsize=8)
        a.set_xticks(range(3)); a.set_xticklabels(CENTERS); a.set_yticks(range(len(M)))
        a.set_yticklabels(["Local " + c for c in CENTERS][:len(M)]); a.set_title(metric + ": train (row) → test (col)"); a.grid(False)
    plt.tight_layout(); plt.show()
    # generalisation gap: own-center score minus score of another center's model
    fg = np.array([B["local%d" % i]["dice_fg"] for i in range(3)])
    display(pd.DataFrame(np.diag(fg)[None, :] - fg, index=["Local " + c for c in CENTERS], columns=CENTERS)
            .round(4).style.set_caption("dice_fg(own local model) − dice_fg(this model), per test center — >0 means a transfer loss"))
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Update geometry under FedAvg — ε calibration and the split CFL *would* make

Alg. 5 splits a cluster when (Eq. 32) ‖mean Δθ‖ < ε₁ (the federation is near a stationary
point) and (Eq. 33) max‖Δθᵢ‖ > ε₂ (some client is still far from *its* optimum). The split
chosen is the bipartition minimising the max cross cosine similarity α (Eq. 51). With 3 clients
there are only 3 bipartitions — {Guys}|{HH,IOP}, {HH}|{Guys,IOP}, {IOP}|{Guys,HH} — and the
chosen one is simply *"isolate the client whose update agrees least with the other two"*.

Quality indicators computed here:
* the three pairwise similarities over time,
* **proposal stability** — fraction of rounds in which each bipartition is the argmin,
* **γ-bound** √((1−α_cross^max)/2): Alg. 5 line 18 only permits a split if the gradient-noise
  bound γ_max is below it — a bound near 0 means the halves *agree* and the split is not
  theoretically justified.
""")

code(r"""
def pair_series(r):
    S = r["S"]; return {f"{CENTERS[i]}–{CENTERS[j]}": S[:, i, j] for i, j in [(0, 1), (0, 2), (1, 2)]}

fa = load_run(RUN_FEDAVG)
if fa:
    m = fa["metrics"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
    ax[0].plot(m["round"], m["mean_norm_max"], label="‖mean Δθ‖ (Eq. 32)")
    ax[0].plot(m["round"], m["max_norm_max"], label="max‖Δθᵢ‖ (Eq. 33)")
    for c in CENTERS: ax[0].plot(m["round"], m["norm_" + c], lw=.8, ls="--", label="‖Δθ‖ " + c)
    ax[0].set_yscale("log"); ax[0].set_title("update norms under FedAvg"); ax[0].legend(fontsize=7); ax[0].set_xlabel("round")
    ratio = m["mean_norm_max"] / m["max_norm_max"]
    ax[1].plot(m["round"], ratio); ax[1].set_title("‖mean Δθ‖ / max‖Δθᵢ‖  (1 = perfectly aligned)"); ax[1].set_xlabel("round")
    for k, v in pair_series(fa).items(): ax[2].plot(np.arange(1, len(v) + 1), v, label=k)
    ax[2].set_title("pairwise cosine similarity α_ij"); ax[2].legend(); ax[2].set_xlabel("round")
    plt.tight_layout(); plt.show()
    tail = m[m["round"] > m["round"].max() * 0.5]
    print("second half of training: ‖mean Δθ‖ in [{:.4f}, {:.4f}], max‖Δθᵢ‖ in [{:.4f}, {:.4f}]".format(
        tail.mean_norm_max.min(), tail.mean_norm_max.max(), tail.max_norm_max.min(), tail.max_norm_max.max()))
    print("=> a CFL split needs ε₁ above the first range and ε₂ below the second; the two ranges "
          "are separated only because clients disagree. Ratio mean/max, 2nd half: {:.3f}".format(ratio[m['round'] > m['round'].max() * .5].mean()))
""")

code(r"""
def proposals(r):
    names = r["setup"]["client_names"]; out = []
    for rec in r["rounds"]:
        st = rec["cluster_stats"][0]
        if st["proposed"] is None: out.append(None); continue
        a, b = st["proposed"]; small = a if len(a) < len(b) else b
        out.append("isolate " + "+".join(names[i] for i in small))
    return out

if fa:
    prop = proposals(fa)
    s = pd.Series(prop)
    display(pd.DataFrame({"all rounds": s.value_counts(normalize=True),
                          "last half": s[len(s) // 2:].value_counts(normalize=True)}).round(3)
            .style.set_caption("bipartition proposed by Alg. 5 line 16 under FedAvg (stability)"))
    st = pd.DataFrame([rec["cluster_stats"][0] for rec in fa["rounds"]])
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    ax[0].plot(st.index + 1, st.alpha_cross_max, label="α_cross^max of proposed split")
    ax[0].plot(st.index + 1, st.alpha_intra_min, label="α_intra^min (the pair kept together)")
    ax[0].plot(st.index + 1, st.gap_est, label="separation gap g = intra − cross")
    ax[0].axhline(0, c="k", lw=.6); ax[0].legend(fontsize=8); ax[0].set_xlabel("round")
    ax[1].plot(st.index + 1, st.gamma_bound); ax[1].set_title("γ bound √((1−α_cross^max)/2)"); ax[1].set_xlabel("round")
    plt.tight_layout(); plt.show()
    print("linkage == exhaustive argmin in {:.0%} of rounds".format(st.linkage_agrees.mean()))
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. The CFL run
""")

code(r"""
cfl = load_run(RUN_CFL)
if cfl:
    f = cfl["final"]; names = cfl["setup"]["client_names"]
    print("split rounds:", f["split_rounds"]); print("final clusters:", f["final_clusters_named"])
    for e in f["split_events"]:
        print("  round {round}: {a} | {b}  mean_norm={mean_norm:.4f} max_norm={max_norm:.4f} α_cross^max={alpha_cross_max:.3f}".format(**e))
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
    for r, ls in [(cfl, "-"), (fa, "--")]:
        if r is None: continue
        m = r["metrics"]
        for c in r["setup"]["client_names"]:
            ax[0].plot(m["round"], m["dice_fg_" + c], ls=ls, label=f"{r['name']}: {c}")
        ax[1].plot(m["round"], m["mean_norm_max"], ls=ls, label=r["name"] + " ‖mean Δθ‖")
        ax[1].plot(m["round"], m["max_norm_max"], ls=ls, label=r["name"] + " max‖Δθ‖")
    for sr in f["split_rounds"]:
        for a in ax: a.axvline(sr, c="r", lw=.8, alpha=.6)
    ax[0].set_title("hard foreground Dice per client"); ax[0].legend(fontsize=6); ax[0].set_ylim(bottom=max(0, ax[0].get_ylim()[0]))
    ax[1].axhline(cfl["cfg"]["eps_1"], c="C0", ls=":", label="ε₁"); ax[1].axhline(cfl["cfg"]["eps_2"], c="C1", ls=":", label="ε₂")
    ax[1].set_yscale("log"); ax[1].legend(fontsize=7); ax[1].set_title("split criteria")
    ax[2].plot(cfl["metrics"]["round"], cfl["metrics"]["n_clusters"]); ax[2].set_title("#clusters")
    plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Oracle — every partition of the 3 clients

All five set partitions are trained with the identical FedAvg protocol (60 rounds,
`--strategy partition`). This is the ground truth that the natural split lacks: the best
partition *for each client* and the best on average. CFL's split is judged against it.

Uncertainty: test sets are small (62 / 36 / **15**), so each difference is reported with a
**paired bootstrap 95 % CI over test subjects** (same subjects, two models).
""")

code(r"""
def boot_ci(a, b, n=5000, seed=0):
    d = np.asarray(a) - np.asarray(b); rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

P = {k: load_run(v) for k, v in PARTITION_RUNS.items()}
P = {k: v for k, v in P.items() if v and v["final"]}
if RUN_CFL and cfl and cfl["final"]:
    P["CFL (" + " ".join("{" + ",".join(c) + "}" for c in cfl["final"]["final_clusters_named"]) + ")"] = cfl
rows = []
for k, r in P.items():
    fg = [r["final"]["cross"]["dice_fg"][c][c] for c in range(3)]
    dd = [r["final"]["cross"]["dice"][c][c] for c in range(3)]
    rows.append({"partition": k, **{"dice_fg " + CENTERS[c]: fg[c] for c in range(3)},
                 "dice_fg mean": np.mean(fg), "FLamby dice mean": np.mean(dd)})
if rows:
    T = pd.DataFrame(rows).set_index("partition")
    display(T.round(4).style.highlight_max(axis=0, color="#cde").set_caption("oracle: final Dice per client under each partition"))
    base = PARTITION_RUNS["{Guys,HH,IOP}"]
    if exists(base):
        fb = load_run(base)["final"]["cross"]["dice_fg_samples"]
        out = []
        for k, r in P.items():
            s = r["final"]["cross"].get("dice_fg_samples")
            if s is None or r["name"] == base: continue
            for c in range(3):
                m, lo, hi = boot_ci(s[c][c], fb[c][c])
                out.append({"partition": k, "client": CENTERS[c], "Δ dice_fg vs FedAvg": m, "CI low": lo, "CI high": hi,
                            "significant": "yes" if lo > 0 or hi < 0 else "no"})
        if out: display(pd.DataFrame(out).set_index(["partition", "client"]).round(4)
                        .style.set_caption("paired bootstrap: partition − FedAvg, per client (95 % CI)"))
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Virtual sub-clients — a split with a known answer

Each center's training set is cut uniformly at random into *k* sub-clients
(`--clients-per-center k`). Sub-clients of one center are i.i.d. with each other, so the
**center is the ground-truth cluster**, exactly like a label permutation in the CIFAR
experiment. We can therefore measure:

* **separation gap** g(α) with true center labels (Eq. 48) — > 0 ⇔ the cosine similarities
  *can* recover the centers (Corollary 1);
* **ARI** of the final CFL partition vs the centers;
* whether splits ever cut *through* a center (a wrong split).
""")

code(r"""
vr = load_run(RUN_VIRTUAL)
if vr:
    names = vr["setup"]["client_names"]; truth = np.array(vr["setup"]["client_center"])
    f = vr["final"]
    print("split rounds:", f["split_rounds"]); print("final clusters:", f["final_clusters_named"])
    lab = np.zeros(len(names), int)
    for k, c in enumerate(f["final_clusters"]): lab[c] = k
    print("ARI(final clusters, center) = {:.3f}".format(adjusted_rand_score(truth, lab)))
    pure = [len(set(truth[c])) == 1 for c in f["final_clusters"]]
    print("pure clusters: {}/{}".format(sum(pure), len(pure)))
    for e in f["split_events"]:
        ok = not (set(truth[e["a"]]) & set(truth[e["b"]]))
        print("  round {}: {} | {} -> {}".format(e["round"], [names[i] for i in e["a"]], [names[i] for i in e["b"]],
                                                "respects centers" if ok else "CUTS THROUGH A CENTER"))
    from cfl_ixi.fl_devices import separation_stats
    S = vr["S"]
    g = [separation_stats(S[t], truth) for t in range(len(S))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    ax[0].plot([x[2] for x in g], label="g(α) with true centers"); ax[0].plot([x[0] for x in g], label="α_intra^min"); ax[0].plot([x[1] for x in g], label="α_cross^max")
    ax[0].axhline(0, c="k", lw=.6); ax[0].legend(fontsize=8); ax[0].set_title("full-population similarities (all clients)")
    for sr in f["split_rounds"]: ax[0].axvline(sr - 1, c="r", lw=.8)
    t0 = min(len(S) - 1, (f["split_rounds"][0] - 1) if f["split_rounds"] else len(S) - 1)
    im = ax[1].imshow(S[t0], cmap="coolwarm", vmin=-1, vmax=1); ax[1].set_title(f"α at round {t0 + 1}"); plt.colorbar(im, ax=ax[1])
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=90, fontsize=7); ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=7); ax[1].grid(False)
    m = vr["metrics"]; ax[2].plot(m["round"], m["dice_fg_mean"], label="mean client"); ax[2].plot(m["round"], m["dice_fg_min"], label="min client")
    ax[2].legend(); ax[2].set_title("dice_fg")
    plt.tight_layout(); plt.show()
    # within- vs between-center similarity distribution over all rounds
    same = truth[:, None] == truth[None, :]; off = ~np.eye(len(truth), dtype=bool)
    w = S[:, same & off].ravel(); b = S[:, ~same].ravel()
    d = (w.mean() - b.mean()) / np.sqrt((w.var() + b.var()) / 2)
    print("mean α within center {:.3f} vs between {:.3f} | Cohen d = {:.2f}".format(w.mean(), b.mean(), d))
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Scorecard
""")

code(r"""
score = []
def add(k, v): score.append({"quantity": k, "value": v})
if B.get("pooled"): add("Pooled dice (FLamby soft) mean over centers", round(float(np.mean(B["pooled"]["dice"])), 4))
for label, name in [("FedAvg FLamby 7 rounds", RUN_FEDAVG_FLAMBY), ("FedAvg 60 rounds", RUN_FEDAVG), ("CFL", RUN_CFL)]:
    r = load_run(name)
    if r and r["final"]:
        add(label + " dice / dice_fg", "{:.4f} / {:.4f}".format(np.mean([r["final"]["cross"]["dice"][c][c] for c in range(3)]),
                                                                 np.mean([r["final"]["cross"]["dice_fg"][c][c] for c in range(3)])))
if cfl and cfl["final"]:
    add("CFL split rounds", cfl["final"]["split_rounds"]); add("CFL final clusters", cfl["final"]["final_clusters_named"])
if rows:
    add("oracle best partition (mean dice_fg)", T["dice_fg mean"].idxmax())
if vr:
    add("virtual: ARI vs centers", round(adjusted_rand_score(truth, lab), 3))
display(pd.DataFrame(score).set_index("quantity"))
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = os.path.join(root, "analysis.ipynb")
    nbf.write(nb, path)
    print("wrote", path)
    if a.execute:
        subprocess.check_call([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                               "--inplace", "--ExecutePreprocessor.timeout=1200", path], cwd=root)
        print("executed", path)


if __name__ == "__main__":
    main()
