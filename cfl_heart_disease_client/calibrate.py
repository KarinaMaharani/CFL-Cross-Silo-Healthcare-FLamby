"""Read eps_1, eps_2 and gamma_max off a finished FedAvg run.

The CFL paper (Sattler et al., Alg. 5) takes eps_1, eps_2 and gamma_max as inputs and
never says how to choose them; they are raw magnitudes that depend on the model, the
learning rate and the number of local steps.  This script applies ONE fixed rule, so
the values are not hand-picked per experiment:

  stationarity  the first round whose Dice gain over the previous ``--window`` rounds
                is below ``--tol``  (FedAvg has stopped improving = Eq. 32's premise)
  eps_1         the weighted mean update norm at that round, rounded up slightly
  eps_2         just below the SMALLEST max_i ||dtheta_i|| observed up to the end
                (so a client that never settles keeps Eq. 33 true)
  gamma_max     median of the last 3 gamma-probe estimates, if the run had a probe

    python -m cfl_heart_disease_client.calibrate --run fedavg18
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .run_experiment import _resolve


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="fedavg8")
    p.add_argument("--results-dir", default="results_heart8")
    p.add_argument("--window", type=int, default=20, help="rounds over which to measure the gain")
    p.add_argument("--tol", type=float, default=0.005, help="accuracy gain that counts as 'flat'")
    p.add_argument("--metric", default="accuracy_mean")
    p.add_argument("--margin", type=float, default=1.03, help="eps_1 = margin x mean norm")
    a = p.parse_args(argv)

    d = os.path.join(_resolve(a.results_dir), a.run)
    m = pd.read_csv(os.path.join(d, "metrics.csv"))
    gain = m[a.metric] - m[a.metric].shift(a.window)
    flat = m.index[(gain < a.tol) & gain.notna()]
    if len(flat) == 0:
        r = int(m.index[-1])
        print("WARNING: {} never flattened within {} rounds; using the last round"
              .format(a.metric, len(m)))
    else:
        r = int(flat[0])

    eps1 = float(m.loc[r, "mean_norm_max"]) * a.margin
    eps2 = float(m["max_norm_max"].min()) * 0.97
    gammas = m["gamma_hat"].dropna().tolist() if "gamma_hat" in m else []
    gamma = float(np.median(gammas[-3:])) if gammas else None

    print("run                        : {}".format(d))
    print("stationary at round        : {}  ({} gain over {} rounds = {:.4f} < {})".format(
        int(m.loc[r, "round"]), a.metric, a.window, float(gain.loc[r]), a.tol))
    print("{:27s}: {:.4f}".format(a.metric + " there", float(m.loc[r, a.metric])))
    print("weighted mean ||dtheta||   : {:.5f}".format(float(m.loc[r, "mean_norm_max"])))
    print("min over rounds of max||dtheta||: {:.5f}".format(float(m["max_norm_max"].min())))
    print("gamma probes (last 3)      : {}".format([round(g, 3) for g in gammas[-3:]]))
    print()
    print("  --eps-1 {:.5f} --eps-2 {:.5f}{}".format(
        eps1, eps2, "" if gamma is None else "   (gamma_hat ~ {:.3f}; use --gamma-auto"
        " --gamma-probe-every 5)".format(gamma)))

    with open(os.path.join(d, "calibration.json"), "w") as f:
        json.dump({"stationary_round": int(m.loc[r, "round"]), "eps_1": eps1, "eps_2": eps2,
                   "gamma_hat_median_last3": gamma, "rule": {
                       "metric": a.metric, "window": a.window, "tol": a.tol,
                       "eps1_margin": a.margin, "eps2_margin": 0.97}}, f, indent=2)
    print("\nwritten to {}".format(os.path.join(d, "calibration.json")))


if __name__ == "__main__":
    main()
