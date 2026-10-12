"""Non-federated reference points for 18-client Fed-ISIC2019.

FLamby's benchmark (fed_isic2019/common.py): Adam, lr 5e-4, batch 64,
NUM_EPOCHS_POOLED = 20 epochs, weighted focal loss, balanced accuracy.

    pooled        one model on all 18 413 training images    -- the upper bound
    center0..5    one model per hospital                     -- what CFL should match
                                                                if it recovers centres
    client0..17   one model per sub-client (117..3310 images) -- the "no federation"
                                                                lower bound

Every model is tested on every centre, giving the 6x6 cross-site matrix.

    python -m cfl_isic_client.baselines                    # pooled + 6 centres + 18 clients
    python -m cfl_isic_client.baselines --only pooled center4 --epochs 20
"""
import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

from .config import Config
from .data_utils import build_clients_data, ClientData, train_transform, SHORT_NAMES
from .fl_devices import autocast_for, resolve_device
from .metrics import focal_loss, evaluate
from .models import model_factory, count_parameters
from .run_experiment import RunLogger, setup_torch, _resolve


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="subset of: pooled center0..5 client0..17")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clients-per-center", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--out", default="results_isic18/baselines")
    p.add_argument("--save-models", action="store_true")
    a = p.parse_args(argv)

    cfg = replace(Config(), seed=a.seed, clients_per_center=a.clients_per_center)
    cfg.data_root = _resolve(cfg.data_root)
    out = _resolve(a.out)
    os.makedirs(out, exist_ok=True)
    log = RunLogger(os.path.join(out, "log.txt"))
    dev = setup_torch(cfg, log)
    autocast = autocast_for(cfg)
    train_data, _, client_center, center_test, _, _ = build_clients_data(cfg, log)
    log("model: FLamby EfficientNet-B0, {} parameters".format(
        count_parameters(model_factory(cfg)())))

    jobs = {"pooled": list(range(len(train_data)))}
    for c in range(len(SHORT_NAMES)):
        jobs["center{}".format(c)] = [i for i in range(len(train_data))
                                      if client_center[i] == c]
    for i in range(len(train_data)):
        jobs["client{}".format(i)] = [i]
    if a.only:
        missing = [k for k in a.only if k not in jobs]
        if missing:
            raise SystemExit("unknown job(s): {}; choose from {}".format(missing, sorted(jobs)))
        jobs = {k: jobs[k] for k in a.only}

    for name, ids in jobs.items():
        torch.manual_seed(a.seed)
        idx = np.concatenate([train_data[i].idx for i in ids])
        lab = torch.cat([train_data[i].labels for i in ids]).numpy()
        data = ClientData(train_data[0].images, idx, lab, cfg.batch_size, a.seed, name, True)
        model = model_factory(cfg)().to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        steps = int(np.ceil(len(data) / cfg.batch_size))
        log("=== {}: clients {} | {} images | {} steps/epoch x {} epochs ===".format(
            name, [train_data[i].name for i in ids], len(data), steps, a.epochs))
        hist = []
        for ep in range(1, a.epochs + 1):
            t0, run_loss = time.time(), 0.0
            model.train()
            for _ in range(steps):
                x, y = data.next_batch()
                x = train_transform(x.to(dev, non_blocking=True), gen)
                x = x.contiguous(memory_format=torch.channels_last)
                y = y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with autocast():
                    o = model(x)
                loss = focal_loss(o, y)
                loss.backward()
                opt.step()
                run_loss += loss.item()
            rec = {"epoch": ep, "train_loss": run_loss / steps, "seconds": time.time() - t0}
            if ep % a.eval_every == 0 or ep == a.epochs:
                r = [evaluate(model, t, cfg, dev, autocast) for t in center_test]
                rec["bacc"] = [e["bacc"] for e in r]
                rec["acc"] = [e["acc"] for e in r]
            hist.append(rec)
            log("{:9s} epoch {:2d}/{} | loss {:.4f} | bacc per centre {} | {:.0f}s".format(
                name, ep, a.epochs, rec["train_loss"],
                [round(v, 3) for v in rec.get("bacc", [])], rec["seconds"]))
        final = hist[-1]
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump({"name": name, "train_clients": ids,
                       "train_client_names": [train_data[i].name for i in ids],
                       "n_train": len(data), "epochs": a.epochs, "lr": a.lr,
                       "seed": a.seed, "test_centers": SHORT_NAMES,
                       "bacc": final["bacc"], "acc": final["acc"],
                       "history": hist}, f, indent=2)
        if a.save_models:
            torch.save(model.state_dict(), os.path.join(out, name + ".pt"))

    summary = {}
    for k in sorted(os.listdir(out)):
        if k.endswith(".json") and k != "summary.json":
            r = json.load(open(os.path.join(out, k)))
            summary[r["name"]] = {"bacc": r["bacc"], "n_train": r["n_train"]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("summary (columns = test centre {}):".format(SHORT_NAMES))
    for k, v in summary.items():
        log("  {:9s} n={:5d} bacc {}".format(k, v["n_train"], np.round(v["bacc"], 3).tolist()))


if __name__ == "__main__":
    main()
