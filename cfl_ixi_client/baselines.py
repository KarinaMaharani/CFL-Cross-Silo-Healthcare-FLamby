"""Non-federated reference points for 9-client Fed-IXI.

FLamby's Fed-IXI benchmark (fed_ixi/benchmark.py, common.py): AdamW, lr 1e-3,
batch 2, NUM_EPOCHS_POOLED = 10 epochs, Dice loss.

    pooled      one model on all 453 training volumes      -- the upper bound
    center0..2  one model per hospital (Guys / HH / IOP)   -- what CFL should match
                                                              if it recovers centres
    client0..8  one model per sub-client (83 / 48 / 20 vol) -- the "no federation at
                                                              all" lower bound, i.e.
                                                              what a split costs when
                                                              it goes all the way

Every model is tested on every centre, giving the cross-site matrix that says how
different the hospitals really are, independently of any CFL decision.

    python -m cfl_ixi_client.baselines                     # pooled + 3 centres + 9 clients
    python -m cfl_ixi_client.baselines --only pooled center0
"""
import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

from .config import Config
from .data_utils import build_clients_data, CENTERS, TensorClientData
from .metrics import dice_loss, one_hot, evaluate
from .models import model_factory, count_parameters
from .run_experiment import RunLogger, setup_torch, _resolve


def train_centralized(x, y, cfg, dev, log, epochs, lr, tag, center_test, eval_every=2):
    torch.manual_seed(cfg.seed)
    data = TensorClientData(x, y, cfg.batch_size, cfg.seed,
                            dev if cfg.pin_data_to_device else None, tag)
    model = model_factory(cfg)().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        t0, running, nb = time.time(), 0.0, 0
        model.train()
        for xb, mb in data.batches(shuffle=True):
            xb, mb = xb.to(dev), mb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = dice_loss(model(xb), one_hot(mb))
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        rec = {"epoch": ep, "train_loss": running / nb, "seconds": time.time() - t0}
        if ep % eval_every == 0 or ep == epochs:
            r = [evaluate(model, t, dev) for t in center_test]
            rec["dice"] = [e["dice"] for e in r]
            rec["dice_fg"] = [e["dice_fg"] for e in r]
            rec["dice_fg_samples"] = [e["dice_fg_samples"] for e in r]
        history.append(rec)
        log("{:9s} epoch {:2d}/{} | loss {:.4f} | dice_fg per centre {} | {:.1f}s".format(
            tag, ep, epochs, rec["train_loss"],
            [round(v, 4) for v in rec.get("dice_fg", [])], rec["seconds"]))
    return model, history


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="subset of: pooled center0..2 client0..8")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clients-per-center", type=int, default=3)
    p.add_argument("--out", default="results_ixi9/baselines")
    p.add_argument("--save-models", action="store_true")
    a = p.parse_args(argv)

    cfg = replace(Config(), seed=a.seed, clients_per_center=a.clients_per_center)
    cfg.data_root, cfg.cache_dir = _resolve(cfg.data_root), _resolve(cfg.cache_dir)
    out = _resolve(a.out)
    os.makedirs(out, exist_ok=True)
    log = RunLogger(os.path.join(out, "log.txt"))
    dev = setup_torch(cfg, log)
    train_data, _, client_center, center_test, _ = build_clients_data(cfg, None, log)
    log("model: FLamby Fed-IXI 3D U-Net, {} parameters".format(
        count_parameters(model_factory(cfg)())))

    jobs = {"pooled": list(range(len(train_data)))}
    for c in range(len(CENTERS)):
        jobs["center{}".format(c)] = [i for i in range(len(train_data))
                                      if client_center[i] == c]
    for i in range(len(train_data)):
        jobs["client{}".format(i)] = [i]
    if a.only:
        missing = [k for k in a.only if k not in jobs]
        if missing:
            raise SystemExit("unknown job(s): {}; choose from {}".format(
                missing, sorted(jobs)))
        jobs = {k: jobs[k] for k in a.only}

    for name, ids in jobs.items():
        x = torch.cat([train_data[i].x for i in ids])
        y = torch.cat([train_data[i].y for i in ids])
        log("=== {}: clients {} ({} volumes), AdamW lr {} x {} epochs ===".format(
            name, [train_data[i].name for i in ids], len(y), a.lr, a.epochs))
        model, hist = train_centralized(x, y, cfg, dev, log, a.epochs, a.lr, name,
                                        center_test)
        final = hist[-1]
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump({"name": name, "train_clients": ids,
                       "train_client_names": [train_data[i].name for i in ids],
                       "n_train": len(y), "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
                       "test_centers": CENTERS,
                       "dice": final["dice"], "dice_fg": final["dice_fg"],
                       "dice_fg_samples": final["dice_fg_samples"],
                       "history": [{k: v for k, v in h.items()
                                    if k != "dice_fg_samples"} for h in hist]}, f, indent=2)
        if a.save_models:
            torch.save(model.state_dict(), os.path.join(out, name + ".pt"))

    summary = {}
    for k in sorted(os.listdir(out)):
        if k.endswith(".json") and k != "summary.json":
            r = json.load(open(os.path.join(out, k)))
            summary[r["name"]] = {"dice": r["dice"], "dice_fg": r["dice_fg"],
                                  "n_train": r["n_train"]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("summary (columns = test centre {}):".format(CENTERS))
    for k, v in summary.items():
        log("  {:9s} n={:3d} dice {} dice_fg {}".format(
            k, v["n_train"], np.round(v["dice"], 4).tolist(),
            np.round(v["dice_fg"], 4).tolist()))


if __name__ == "__main__":
    main()
