"""Non-federated reference points for 18-client Fed-Heart-Disease.

FLamby's benchmark (fed_heart_disease/common.py): Adam, lr 1e-3, batch 4,
NUM_EPOCHS_POOLED = 50 epochs, binary cross-entropy, accuracy.

    pooled      one model on all ~486 training patients  -- the upper bound
    center0..3  one model per hospital                    -- what CFL should match
                                                           if it recovers regions
    client0..7  one model per sub-client (~15..100 patients) -- the "no federation"
                                                             lower bound

Every model is evaluated on every hospital's test set (the 4x4 cross-site matrix).

    python -m cfl_heart_disease_client.baselines
    python -m cfl_heart_disease_client.baselines --only pooled center5
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
from .models import bce_loss, evaluate, model_factory, count_parameters
from .run_experiment import RunLogger, setup_torch, _resolve


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="subset of: pooled center0..3 client0..7")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clients-per-center", type=int, default=2)
    p.add_argument("--out", default="results_heart8/baselines")
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
    for c in range(4):
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
        log("=== {}: clients {} | {} patients (pos {:.2f}) | {} steps/epoch x {} ===".format(
            name, [names[i] for i in ids], len(data), float(y.mean()), steps, a.epochs))
        hist = []
        for ep in range(1, a.epochs + 1):
            t0, run_loss, used = time.time(), 0.0, 0
            model.train()
            for _ in range(steps):
                xb, yb = data.next_batch()
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = bce_loss(model(xb), yb)
                loss.backward()
                opt.step()
                run_loss += float(loss)
                used += 1
            r = [evaluate(model, t, dev) for t in center_test]
            rec = {"epoch": ep, "train_loss": run_loss / max(used, 1),
                   "accuracy": [e["accuracy"] for e in r], "seconds": time.time() - t0}
            hist.append(rec)
            if ep % 10 == 0 or ep == a.epochs:
                log("{:9s} epoch {:2d}/{} | loss {:.4f} | accuracy per region {}".format(
                    name, ep, a.epochs, rec["train_loss"],
                    [round(v, 3) for v in rec["accuracy"]]))
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump({"name": name, "train_clients": ids,
                       "train_client_names": [names[i] for i in ids],
                       "n_train": len(data), "positive_rate": float(y.mean()),
                       "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
                       "accuracy": hist[-1]["accuracy"], "history": hist}, f, indent=2)

    summary = {}
    for k in sorted(os.listdir(out)):
        if k.endswith(".json") and k != "summary.json":
            r = json.load(open(os.path.join(out, k)))
            summary[r["name"]] = {"accuracy": r["accuracy"], "n_train": r["n_train"],
                                  "positive_rate": r["positive_rate"]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("summary (columns = test hospital 0..3):")
    for k, v in summary.items():
        log("  {:9s} n={:3d} (pos {:.2f}) accuracy {}".format(
            k, v["n_train"], v["positive_rate"], np.round(v["accuracy"], 3).tolist()))


if __name__ == "__main__":
    main()
