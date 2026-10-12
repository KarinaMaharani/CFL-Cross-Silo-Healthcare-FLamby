"""Non-federated reference points for 18-client Fed-TCGA-BRCA.

FLamby's benchmark (fed_tcga_brca/common.py): Adam, lr 1e-1, batch 8,
NUM_EPOCHS_POOLED = 30 epochs, Cox partial likelihood, concordance index.

    pooled      one model on all 866 training patients  -- the upper bound
    center0..5  one model per region                    -- what CFL should match
                                                           if it recovers regions
    client0..17 one model per sub-client (13..83 patients) -- the "no federation"
                                                             lower bound

Every model is evaluated on every region's test set (the 6x6 cross-site matrix).

    python -m cfl_tcga_brca_client.baselines
    python -m cfl_tcga_brca_client.baselines --only pooled center5
"""
import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

from .config import Config
from .data_utils import build_clients_data, TensorClientData
from .models import cox_loss, evaluate, model_factory, count_parameters
from .run_experiment import RunLogger, setup_torch, _resolve


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="subset of: pooled center0..5 client0..17")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clients-per-center", type=int, default=3)
    p.add_argument("--out", default="results_brca18/baselines")
    a = p.parse_args(argv)

    cfg = replace(Config(), seed=a.seed, clients_per_center=a.clients_per_center)
    cfg.data_root = _resolve(cfg.data_root)
    out = _resolve(a.out)
    os.makedirs(out, exist_ok=True)
    log = RunLogger(os.path.join(out, "log.txt"))
    dev = setup_torch(cfg, log)
    train_data, _, client_center, center_test, meta = build_clients_data(cfg, dev, log)
    names = meta["names"]
    log("model: FLamby Cox linear model, {} parameters".format(
        count_parameters(model_factory(cfg)())))

    jobs = {"pooled": list(range(len(train_data)))}
    for c in range(6):
        jobs["center{}".format(c)] = [i for i in range(len(train_data))
                                      if client_center[i] == c]
    for i in range(len(train_data)):
        jobs["client{}".format(i)] = [i]
    if a.only:
        bad = [k for k in a.only if k not in jobs]
        if bad:
            raise SystemExit("unknown job(s): {}".format(bad))
        jobs = {k: jobs[k] for k in a.only}

    for name, ids in jobs.items():
        torch.manual_seed(a.seed)
        x = torch.cat([train_data[i].x for i in ids])
        y = torch.cat([train_data[i].y for i in ids])
        data = TensorClientData(x, y, cfg.batch_size, a.seed, name)
        model = model_factory(cfg)().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        steps = int(np.ceil(len(data) / cfg.batch_size))
        log("=== {}: clients {} | {} patients ({:.0f} events) | {} steps/epoch x {} ===".format(
            name, [names[i] for i in ids], len(data), float(y[:, 0].sum()), steps, a.epochs))
        hist = []
        for ep in range(1, a.epochs + 1):
            t0, run_loss, used = time.time(), 0.0, 0
            model.train()
            for _ in range(steps):
                xb, yb = data.next_batch()
                xb, yb = xb.to(dev), yb.to(dev)
                if len(yb) < 2 or float(yb[:, 0].sum()) == 0:
                    continue
                opt.zero_grad(set_to_none=True)
                loss = cox_loss(model(xb), yb)
                loss.backward()
                opt.step()
                run_loss += float(loss)
                used += 1
            r = [evaluate(model, t, dev) for t in center_test]
            rec = {"epoch": ep, "train_loss": run_loss / max(used, 1),
                   "c_index": [e["c_index"] for e in r], "seconds": time.time() - t0}
            hist.append(rec)
            if ep % 10 == 0 or ep == a.epochs:
                log("{:9s} epoch {:2d}/{} | loss {:.4f} | c-index per region {}".format(
                    name, ep, a.epochs, rec["train_loss"],
                    [round(v, 3) for v in rec["c_index"]]))
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump({"name": name, "train_clients": ids,
                       "train_client_names": [names[i] for i in ids],
                       "n_train": len(data), "n_events": int(y[:, 0].sum()),
                       "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
                       "c_index": hist[-1]["c_index"], "history": hist}, f, indent=2)

    summary = {}
    for k in sorted(os.listdir(out)):
        if k.endswith(".json") and k != "summary.json":
            r = json.load(open(os.path.join(out, k)))
            summary[r["name"]] = {"c_index": r["c_index"], "n_train": r["n_train"],
                                  "n_events": r["n_events"]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("summary (columns = test region 0..5):")
    for k, v in summary.items():
        log("  {:9s} n={:3d} ({:2d} events) c-index {}".format(
            k, v["n_train"], v["n_events"], np.round(v["c_index"], 3).tolist()))


if __name__ == "__main__":
    main()
