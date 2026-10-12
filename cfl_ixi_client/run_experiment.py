"""Clustered Federated Learning on 9-client Fed-IXI: FedAvg, CFL, fixed partitions.

    # plain FedAvg (never splits) -- the reference AND the eps calibration run
    python -m cfl_ixi_client.run_experiment --run-name fedavg9 --strategy fedavg --rounds 60

    # CFL with the paper's criteria only
    python -m cfl_ixi_client.run_experiment --run-name cfl9 --strategy cfl \
        --eps-1 <e1> --eps-2 <e2> --gamma-auto --gamma-probe-every 5

    # with stragglers: every centre has a 1.0 / 0.5 / 0.25-speed client
    python -m cfl_ixi_client.run_experiment --run-name cfl9_strag --strategy cfl \
        --straggler-mode partial --straggler-profile tiers --straggler-tiers 1.0,0.5,0.25

    # oracle: the ground-truth partition (centres) trained from round 1
    python -m cfl_ixi_client.run_experiment --run-name oracle_centers --strategy partition \
        --partition "0,1,2|3,4,5|6,7,8"

Outputs land in results_ixi9/<run_name>/:
    config.json        exact hyper-parameters
    setup.json         client -> centre map, sizes, the scan indices of every client
    log.txt            human-readable log (every split decision and rejection)
    metrics.csv        flat per-round table (Dice, ARI, norms, gaps, stragglers)
    rounds.jsonl       one JSON per round: per-client Dice/loss/norms/steps + cluster stats
    similarities.npz   the n x n cosine-similarity matrix of every round
    cross_eval.jsonl   every client model x every centre test set, every k rounds
    final.json         final clusters, ARI, split events, per-subject Dice
    models/            final client state_dicts (--no-save-models to skip)
"""
import argparse
import csv
import json
import os
import time
from dataclasses import replace

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from .config import Config
from .data_utils import build_clients_data, CENTERS
from .fl_devices import (Client, Server, copy_, pairwise_angles, separation_stats,
                         gamma_bound, optimal_bipartition, complete_linkage_bipartition,
                         resolve_device)
from .models import model_factory, count_parameters
from .stragglers import StragglerSchedule

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path):
    """Relative paths are read against the project root, not the cwd, so the module
    runs the same from anywhere (handy inside tmux on a server)."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


class RunLogger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = "[{}] {}".format(time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def parse_partition(spec, n):
    groups = [np.array([int(i) for i in g.split(",")], dtype=int) for g in spec.split("|")]
    flat = sorted(np.concatenate(groups).tolist())
    if flat != list(range(n)):
        raise ValueError("partition '{}' must cover clients 0..{} exactly once".format(
            spec, n - 1))
    return groups


def setup_torch(cfg, log):
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = resolve_device(cfg.device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        # a replication study needs repeatable numbers: cuDNN autotuning would
        # perturb the very update norms eps_1 / eps_2 are calibrated against
        torch.backends.cudnn.deterministic = cfg.deterministic
        torch.backends.cudnn.benchmark = not cfg.deterministic
        log("device: cuda -- {} ({:.1f} GB), torch {}".format(
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9, torch.__version__))
    else:
        log("device: cpu ({} threads), torch {}".format(cfg.torch_threads, torch.__version__))
    return dev


def cross_evaluate(clients, center_test, samples=False):
    """Every client model on every centre's test set."""
    dice, fg, smp = [], [], []
    for c in clients:
        r = [c.evaluate(t) for t in center_test]
        dice.append([e["dice"] for e in r])
        fg.append([e["dice_fg"] for e in r])
        smp.append([e["dice_fg_samples"] for e in r])
    out = {"dice": dice, "dice_fg": fg}
    if samples:
        out["dice_fg_samples"] = smp
    return out


def cluster_labels(clusters, n):
    lab = np.zeros(n, dtype=int)
    for k, c in enumerate(clusters):
        lab[np.asarray(c, dtype=int)] = k
    return lab


def run(cfg):
    cfg.data_root, cfg.cache_dir = _resolve(cfg.data_root), _resolve(cfg.cache_dir)
    cfg.results_dir = _resolve(cfg.results_dir)
    out = os.path.join(cfg.results_dir, cfg.run_name)
    os.makedirs(out, exist_ok=True)
    log = RunLogger(os.path.join(out, "log.txt"))
    cfg.to_json(os.path.join(out, "config.json"))
    log("=== CFL / Fed-IXI, {} clients | run '{}' | strategy {} ===".format(
        cfg.n_clients(), cfg.run_name, cfg.strategy))
    dev = setup_torch(cfg, log)

    train_data, test_data, client_center, center_test, _ = build_clients_data(cfg, dev, log)
    n = len(train_data)
    names = [d.name for d in train_data]
    make_model = model_factory(cfg)
    clients = [Client(make_model, cfg, train_data[i], test_data[i], i) for i in range(n)]
    server = Server(make_model)
    for c in clients:
        c.synchronize_with_server(server)
    log("model: FLamby Fed-IXI 3D U-Net, {} parameters".format(
        count_parameters(server.model)))
    log("test sets: " + ", ".join("{}={}".format(t.name, len(t)) for t in center_test))

    sched = StragglerSchedule(cfg, n, cfg.clients_per_center)
    log(sched.describe())

    with open(os.path.join(out, "setup.json"), "w") as f:
        json.dump({"client_names": names, "client_center": client_center.tolist(),
                   "centers": CENTERS, "clients_per_center": cfg.clients_per_center,
                   "n_train": [len(d) for d in train_data],
                   "n_test": [len(d) for d in test_data],
                   "n_test_center": [len(t) for t in center_test],
                   "train_indices": [d.indices for d in train_data],
                   "straggler_speeds": sched.speeds(0).tolist(),
                   "ground_truth_clusters": client_center.tolist()}, f, indent=2)

    # ---------------------------------------------------------------- gamma probe
    # Two disjoint halves of ONE client share its distribution, so any dissimilarity
    # between their updates is pure approximation noise.  With the paper's
    # intra-cluster bound alpha >= 1 - 2 gamma^2 this gives a data-driven estimate
    # gamma_hat = sqrt((1 - alpha_halves) / 2) for Alg. 5 line 18.
    probes, gamma_hats = [], []
    if cfg.gamma_probe_every:
        from .data_utils import TensorClientData, partition_pool
        p = cfg.gamma_probe_client
        src_data = train_data[p]
        rng = np.random.default_rng([cfg.seed, 999])
        halves = partition_pool(np.arange(len(src_data)), 2, rng, "array_split")
        for h, part in enumerate(halves):
            idx = torch.from_numpy(np.asarray(part))
            half = TensorClientData(src_data.x[idx], src_data.y[idx], cfg.batch_size,
                                    cfg.seed + 500 + h, None,
                                    "{}-half{}".format(names[p], h))
            probes.append(Client(make_model, cfg, half, test_data[p], 100 + h))
        log("gamma probe: client {} -> halves of {} / {} volumes, every {} rounds{}".format(
            names[p], len(probes[0].train_data), len(probes[1].train_data),
            cfg.gamma_probe_every, " (gamma_max automatic)" if cfg.gamma_auto else ""))

    # ------------------------------------------------------------------- clusters
    if cfg.strategy == "partition":
        clusters = parse_partition(cfg.partition, n)
    else:
        clusters = [np.arange(n).astype(int)]
    born = {tuple(i.tolist()): 0 for i in clusters}

    jsonl = open(os.path.join(out, "rounds.jsonl"), "a", encoding="utf-8")
    xjsonl = open(os.path.join(out, "cross_eval.jsonl"), "a", encoding="utf-8")
    csvf = open(os.path.join(out, "metrics.csv"), "a", newline="", encoding="utf-8")
    writer, header_written = csv.writer(csvf), False
    all_S, split_rounds, split_events = [], [], []
    ev = [{"dice": float("nan"), "dice_fg": float("nan")}] * n
    cum_time, t_start = 0.0, time.time()

    for rnd in range(1, cfg.communication_rounds + 1):
        t0 = time.time()

        # --- gamma probe (before the clients move, from the probe client's weights)
        alpha_halves = gamma_hat = float("nan")
        if probes and (rnd == 1 or rnd % cfg.gamma_probe_every == 0):
            src = clients[cfg.gamma_probe_client]
            for pc in probes:
                copy_(target=pc.W, source=src.W)
                copy_(target=pc.B, source=src.B)
                pc.optimizer.load_state_dict(src.optimizer.state_dict())
                pc.compute_weight_update(cfg.local_updates)
            alpha_halves = float(pairwise_angles([probes[0].dW, probes[1].dW])[0, 1])
            gamma_hat = float(np.sqrt(max(0.0, (1.0 - alpha_halves) / 2.0)))
            gamma_hats.append(gamma_hat)
            log("round {}: gamma probe alpha_halves={:.3f} gamma_hat={:.3f}".format(
                rnd, alpha_halves, gamma_hat))
        gamma_used = (float(np.median(gamma_hats[-3:])) if (cfg.gamma_auto and gamma_hats)
                      else cfg.gamma_max)

        # --- stragglers decide how much local work each client does this round
        plan = sched.plan(rnd, cfg.local_updates)
        steps, dropped = plan["steps"], np.array(plan["dropped"])
        participating = ~dropped
        losses = [clients[i].compute_weight_update(steps[i]) for i in range(n)]
        for c in clients:
            c.reset()
        cum_time += plan["round_time"]

        # --- cosine similarities over the clients that actually reported
        S = np.full((n, n), np.nan)
        part_idx = np.where(participating)[0]
        if len(part_idx) > 1:
            Sp = pairwise_angles([clients[i].dW for i in part_idx])
            S[np.ix_(part_idx, part_idx)] = Sp
        all_S.append(S)
        norms = [float(torch.norm(torch.cat([v.flatten() for v in c.dW.values()])).item())
                 for c in clients]

        new_clusters, records, n_split = [], [], 0
        for idc in clusters:
            members = [clients[i] for i in idc]
            contrib = [clients[i] for i in idc if participating[i]]
            all_in = len(contrib) == len(idc)
            mean_norm = server.compute_mean_update_norm(contrib, cfg.weighted_mean_norm)
            max_norm = server.compute_max_update_norm(contrib)
            Ssub = S[np.ix_(idc, idc)]
            _, _, gap_true = separation_stats(Ssub, client_center[idc]) if all_in else (
                float("nan"), float("nan"), float("nan"))

            intra_min = cross_max = gap_est = float("nan")
            c1 = c2 = None
            linkage_agrees = None
            if len(idc) > 1 and all_in:
                a, b, _ = optimal_bipartition(Ssub)
                la, lb = complete_linkage_bipartition(Ssub)
                linkage_agrees = bool({tuple(sorted(a)), tuple(sorted(b))} ==
                                      {tuple(sorted(la)), tuple(sorted(lb))})
                lbl = np.zeros(len(idc), int)
                lbl[b] = 1
                intra_min, cross_max, gap_est = separation_stats(Ssub, lbl)
                c1, c2 = idc[a], idc[b]
            bound = gamma_bound(cross_max)

            do_split = False
            if cfg.strategy == "cfl" and len(idc) > cfg.min_cluster_size and all_in:
                if cfg.split_schedule:
                    do_split = rnd in cfg.split_schedule
                else:
                    age = rnd - born.get(tuple(idc.tolist()), 0)
                    do_split = (mean_norm < cfg.eps_1 and max_norm > cfg.eps_2
                                and rnd >= cfg.min_split_round
                                and age >= cfg.split_cooldown)
                if do_split and cfg.max_clusters is not None:
                    do_split = len(clusters) + n_split < cfg.max_clusters
                if do_split and gamma_used is not None:
                    criteria_met, do_split = True, (bound == bound and gamma_used < bound)
                    if criteria_met and not do_split:
                        log("round {}: split of {} REJECTED by Alg.5 l.18: "
                            "gamma_max={:.3f} >= bound={:.3f} (alpha_cross_max={:.3f})"
                            .format(rnd, idc.tolist(), gamma_used, bound, cross_max))

            records.append({"members": idc.tolist(), "mean_norm": mean_norm,
                            "max_norm": max_norm, "gap_true": gap_true, "gap_est": gap_est,
                            "alpha_intra_min": intra_min, "alpha_cross_max": cross_max,
                            "gamma_bound": bound, "gamma_used": gamma_used,
                            "all_participated": all_in, "linkage_agrees": linkage_agrees,
                            "proposed": [c1.tolist(), c2.tolist()] if c1 is not None else None,
                            "split": bool(do_split)})

            if do_split and c1 is not None and len(c1) and len(c2):
                new_clusters += [c1, c2]
                born[tuple(c1.tolist())] = born[tuple(c2.tolist())] = rnd
                n_split += 1
                split_rounds.append(rnd)
                pure = (len(set(client_center[c1])) == 1 and len(set(client_center[c2])) == 1)
                split_events.append({"round": rnd, "parent": idc.tolist(),
                                     "a": c1.tolist(), "b": c2.tolist(),
                                     "a_names": [names[i] for i in c1],
                                     "b_names": [names[i] for i in c2],
                                     "mean_norm": mean_norm, "max_norm": max_norm,
                                     "gap_est": gap_est, "alpha_cross_max": cross_max,
                                     "gamma_used": gamma_used, "gamma_bound": bound,
                                     "respects_centers": bool(
                                         not set(client_center[c1]) & set(client_center[c2])),
                                     "both_halves_pure": bool(pure)})
                log("round {}: SPLIT {} -> {} | {}  ({} | {})  mean={:.4f} max={:.4f} "
                    "a_cross={:.3f} g_est={:.3f} gamma={} < bound={:.3f} | {}".format(
                        rnd, idc.tolist(), c1.tolist(), c2.tolist(),
                        "+".join(names[i] for i in c1), "+".join(names[i] for i in c2),
                        mean_norm, max_norm, cross_max, gap_est,
                        "off" if gamma_used is None else "{:.3f}".format(gamma_used), bound,
                        "respects centres" if split_events[-1]["respects_centers"]
                        else "CUTS THROUGH A CENTRE"))
            else:
                new_clusters.append(idc)

        clusters = new_clusters
        server.aggregate_clusterwise([[clients[i] for i in idc] for idc in clusters],
                                     weighted=cfg.weighted_aggregation,
                                     bn_stats=cfg.bn_stats, participating=participating)

        # ------------------------------------------------------------- evaluation
        if rnd % cfg.eval_every == 0 or rnd == cfg.communication_rounds:
            ev = [c.evaluate() for c in clients]
        if cfg.cross_eval_every and (rnd % cfg.cross_eval_every == 0
                                     or rnd == cfg.communication_rounds
                                     or rnd in split_rounds):
            xjsonl.write(json.dumps({"round": rnd,
                                     **cross_evaluate(clients, center_test)}) + "\n")
            xjsonl.flush()
        dt = time.time() - t0

        def _mx(key):
            vals = [r[key] for r in records if r[key] == r[key]]
            return max(vals) if vals else float("nan")

        dice = np.array([e["dice"] for e in ev])
        fg = np.array([e["dice_fg"] for e in ev])
        ari = adjusted_rand_score(client_center, cluster_labels(clusters, n))
        pure = sum(len(set(client_center[np.asarray(c)])) == 1 for c in clusters)
        valid_losses = [v for v in losses if v == v]
        row = {"round": rnd, "n_clusters": len(clusters), "ari": float(ari),
               "pure_clusters": int(pure),
               "dice_mean": float(dice.mean()), "dice_fg_mean": float(fg.mean()),
               "dice_fg_min": float(fg.min()), "dice_fg_max": float(fg.max()),
               "train_loss": float(np.mean(valid_losses)) if valid_losses else float("nan"),
               "mean_norm_max": _mx("mean_norm"), "max_norm_max": _mx("max_norm"),
               "gap_est_max": _mx("gap_est"), "gap_true_max": _mx("gap_true"),
               "alpha_cross_max": _mx("alpha_cross_max"),
               "alpha_halves": alpha_halves, "gamma_hat": gamma_hat,
               "gamma_used": float("nan") if gamma_used is None else gamma_used,
               "n_splits_this_round": n_split,
               "n_dropped": int(dropped.sum()), "steps_min": int(min(steps)),
               "steps_max": int(max(steps)), "round_time": plan["round_time"],
               "cum_round_time": cum_time, "seconds": dt}
        if not header_written:
            writer.writerow(list(row)
                            + ["dice_fg_" + x for x in names]
                            + ["norm_" + x for x in names]
                            + ["steps_" + x for x in names])
            header_written = True
        writer.writerow(list(row.values()) + fg.tolist() + norms + list(steps))
        csvf.flush()
        jsonl.write(json.dumps({**row, "dice_clients": dice.tolist(),
                                "dice_fg_clients": fg.tolist(), "loss_clients": losses,
                                "norm_clients": norms, "steps_clients": list(steps),
                                "dropped_clients": dropped.tolist(),
                                "speeds": plan["speeds"],
                                "clusters": [i.tolist() for i in clusters],
                                "cluster_stats": records}) + "\n")
        jsonl.flush()

        if rnd % cfg.log_every == 0 or rnd == 1:
            eta = (time.time() - t_start) / rnd * (cfg.communication_rounds - rnd)
            log("round {:3d}/{} | dice {:.4f} | dice_fg {:.4f} (min {:.4f}) | loss {:.4f} | "
                "clusters {} (ARI {:.3f}, {} pure) | ||mean dW|| {:.4f} max {:.4f} | "
                "steps {}-{}{} | {:.1f}s | ETA {:.0f}m".format(
                    rnd, cfg.communication_rounds, row["dice_mean"], row["dice_fg_mean"],
                    row["dice_fg_min"], row["train_loss"], len(clusters), ari, pure,
                    row["mean_norm_max"], row["max_norm_max"], row["steps_min"],
                    row["steps_max"],
                    " ({} dropped)".format(row["n_dropped"]) if row["n_dropped"] else "",
                    dt, eta / 60))

    # ----------------------------------------------------------------- wrap up
    final_cross = cross_evaluate(clients, center_test, samples=True)
    lab = cluster_labels(clusters, n)
    final = {"final_clusters": [i.tolist() for i in clusters],
             "final_clusters_named": [[names[j] for j in i] for i in clusters],
             "ari_vs_centers": float(adjusted_rand_score(client_center, lab)),
             "pure_clusters": int(sum(len(set(client_center[np.asarray(c)])) == 1
                                      for c in clusters)),
             "split_rounds": split_rounds, "split_events": split_events,
             "client_names": names, "client_center": client_center.tolist(),
             "per_client": ev, "cross": final_cross,
             "straggler_speeds": sched.speeds(0).tolist(),
             "cum_round_time": cum_time,
             "minutes": (time.time() - t_start) / 60}
    with open(os.path.join(out, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    if cfg.save_similarities:
        np.savez_compressed(os.path.join(out, "similarities.npz"), S=np.stack(all_S))
    if cfg.save_models:
        os.makedirs(os.path.join(out, "models"), exist_ok=True)
        for c in clients:
            torch.save(c.model.state_dict(),
                       os.path.join(out, "models", "client_{}.pt".format(c.id)))

    log("done in {:.1f} min | dice {:.4f} | dice_fg {:.4f} | ARI {:.3f} | clusters {} | "
        "splits at {}".format((time.time() - t_start) / 60,
                              float(np.mean([e["dice"] for e in ev])),
                              float(np.mean([e["dice_fg"] for e in ev])),
                              final["ari_vs_centers"],
                              final["final_clusters_named"], split_rounds))
    for fh in (jsonl, xjsonl, csvf):
        fh.close()
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name")
    p.add_argument("--strategy", choices=["cfl", "fedavg", "partition"])
    p.add_argument("--partition", help='e.g. "0,1,2|3,4,5|6,7,8"')
    p.add_argument("--rounds", type=int, dest="communication_rounds")
    p.add_argument("--local-updates", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--momentum", type=float)
    p.add_argument("--lr-decay", type=float)
    p.add_argument("--grad-clip", type=float, help="<=0 disables")
    p.add_argument("--unweighted", dest="weighted_aggregation", action="store_const",
                   const=False)
    p.add_argument("--bn-stats", choices=["local", "average"])
    # partition
    p.add_argument("--clients-per-center", type=int)
    p.add_argument("--centers")
    p.add_argument("--partition-mode", choices=["array_split", "equal_drop"])
    p.add_argument("--partition-seed-mode", choices=["per_center", "global"])
    p.add_argument("--split-test-set", action="store_const", const=True)
    # CFL criteria
    p.add_argument("--eps-1", type=float)
    p.add_argument("--eps-2", type=float)
    p.add_argument("--unweighted-mean-norm", dest="weighted_mean_norm",
                   action="store_const", const=False)
    p.add_argument("--min-split-round", type=int)
    p.add_argument("--min-cluster-size", type=int)
    p.add_argument("--split-cooldown", type=int)
    p.add_argument("--max-clusters", type=int)
    p.add_argument("--split-schedule", help="comma separated rounds, overrides eps")
    p.add_argument("--gamma-max", type=float)
    p.add_argument("--gamma-auto", action="store_const", const=True)
    p.add_argument("--gamma-probe-every", type=int)
    p.add_argument("--gamma-probe-client", type=int)
    # stragglers
    p.add_argument("--straggler-mode", choices=["none", "partial", "drop"])
    p.add_argument("--straggler-profile", choices=["tiers", "lognormal", "uniform", "manual"])
    p.add_argument("--straggler-speeds", help="manual profile: one speed per client")
    p.add_argument("--straggler-tiers", help="cycled within each centre, e.g. 1.0,0.5,0.25")
    p.add_argument("--straggler-sigma", type=float)
    p.add_argument("--straggler-dynamic", action="store_const", const=True)
    p.add_argument("--straggler-deadline", type=float)
    p.add_argument("--straggler-min-updates", type=int)
    p.add_argument("--straggler-seed", type=int)
    # model (overrides of FLamby's U-Net, for ablations)
    p.add_argument("--f-maps", type=int)
    p.add_argument("--num-levels", type=int)
    p.add_argument("--layer-order")
    p.add_argument("--upsample", choices=["nearest", "trilinear", "deconv"])
    p.add_argument("--dropout-prob", type=float)
    # io
    p.add_argument("--eval-every", type=int)
    p.add_argument("--cross-eval-every", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.add_argument("--results-dir")
    p.add_argument("--no-save-models", dest="save_models", action="store_const", const=False)
    p.add_argument("--no-pin-data", dest="pin_data_to_device", action="store_const",
                   const=False)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_const",
                   const=False)
    a = vars(p.parse_args(argv))
    sched = a.pop("split_schedule", None)
    kw = {k: v for k, v in a.items() if v is not None}
    if kw.get("grad_clip", 1) <= 0:
        kw["grad_clip"] = None
    if sched:
        kw["split_schedule"] = tuple(int(s) for s in sched.split(","))
    return replace(Config(), **kw)


if __name__ == "__main__":
    run(parse_args())
