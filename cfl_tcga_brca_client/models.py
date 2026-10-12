"""The Fed-TCGA-BRCA baseline model, loss and metric, following FLamby exactly.

model  (fed_tcga_brca/model.py)  : a single linear layer, 39 -> 1, no activation.
                                   The output is a Cox *risk score*: higher means
                                   death sooner.  40 parameters in total.
loss   (fed_tcga_brca/loss.py)   : the Cox partial likelihood.
metric (fed_tcga_brca/metric.py) : Harrell's concordance index (c-index), which FLamby
                                   takes from ``lifelines``.  Re-implemented here so
                                   the project keeps one less dependency; the
                                   definition is the standard one and is unit-tested
                                   against hand-computed cases in __main__.

Why this dataset is interesting for CFL: with only 40 parameters the cosine
similarities of Eq. 51 live in a 40-dimensional space, where random directions are NOT
nearly orthogonal.  The approximation noise that swamped the signal on Fed-ISIC2019
(4 M parameters, gamma_hat ~ 0.6) is far smaller here, so the paper's assumptions are
much closer to being satisfied.
"""
import numpy as np
import torch
import torch.nn as nn


class Baseline(nn.Module):
    """FLamby's fed_tcga_brca Baseline: Linear(39, 1), no activation."""

    def __init__(self, input_size=39):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


def build_model(cfg=None):
    return Baseline()


def model_factory(cfg=None):
    return lambda: build_model(cfg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------- loss
def cox_loss(scores, truth):
    """FLamby BaselineLoss: the Cox partial likelihood.

    ``scores`` (n, 1) risk scores, ``truth`` (n, 2) = (E, T).  For every patient i who
    actually died, the loss adds log sum_{j in risk set} exp(score_j - score_i), where
    the risk set is everyone still at risk at i's death.  Minimising it pushes the
    scores of patients who died early above those who survived longer.

    FLamby sorts by DESCENDING time and accumulates a prefix log-sum-exp; the same is
    done here, with the max subtracted for numerical stability exactly as they do.
    """
    scores = scores.squeeze(dim=1)
    if scores.size(0) < 2 or float(truth[:, 0].sum()) == 0:
        # no comparable pair (or no observed event) -> the partial likelihood is empty.
        # Return a differentiable zero so .backward() still works on such a batch.
        return scores.sum() * 0.0
    events, times = truth[:, 0], truth[:, 1]
    order = torch.argsort(-times)                 # reverse-sorted in time, as in FLamby
    scores, events = scores[order], events[order]
    loss = torch.zeros_like(scores)
    for i in range(1, scores.size(0)):
        aux = scores[: i + 1] - scores[i]
        m = aux.max()
        loss[i] = m + torch.log((aux - m).exp().sum(0))
    loss = loss * events
    return loss.mean()


# -------------------------------------------------------------------- metric
def concordance_index(times, scores, events):
    """Harrell's c-index: the share of comparable pairs ranked correctly.

    A pair (i, j) is comparable when the earlier of the two is an observed event.
    The model ranks it correctly when the patient who died earlier has the higher
    risk score.  Ties in score count as half.  This matches
    ``lifelines.utils.concordance_index(times, -scores, events)``, which is what
    FLamby calls (note the minus: lifelines expects a predicted *time*, FLamby passes
    a risk score).
    """
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=float)
    n = len(times)
    num = den = 0.0
    for i in range(n):
        if events[i] != 1:
            continue                              # i must be an observed event
        later = times > times[i]                  # everyone who outlived patient i
        if not later.any():
            continue
        den += later.sum()
        num += (scores[later] < scores[i]).sum() + 0.5 * (scores[later] == scores[i]).sum()
    return float(num / den) if den > 0 else float("nan")


@torch.no_grad()
def evaluate(model, data, device):
    """-> {'c_index', 'loss', 'n', 'n_events'} for one client dataset."""
    was = model.training
    model.eval()
    x, y = data.all()
    x, y = x.to(device), y.to(device)
    scores = model(x)
    loss = float(cox_loss(scores, y))
    ci = concordance_index(y[:, 1].cpu().numpy(), scores.cpu().numpy(), y[:, 0].cpu().numpy())
    model.train(was)
    return {"c_index": ci, "loss": loss, "n": int(len(y)), "n_events": int(y[:, 0].sum())}


if __name__ == "__main__":
    m = Baseline()
    print("Fed-TCGA-BRCA Baseline: {} parameters, out {}".format(
        count_parameters(m), tuple(m(torch.randn(4, 39)).shape)))
    # c-index sanity checks
    t = [1.0, 2.0, 3.0, 4.0]
    e = [1, 1, 1, 0]
    perfect = [4.0, 3.0, 2.0, 1.0]        # earlier death -> higher risk
    worst = [1.0, 2.0, 3.0, 4.0]
    print("perfect ranking c-index:", concordance_index(t, perfect, e))   # 1.0
    print("worst   ranking c-index:", concordance_index(t, worst, e))     # 0.0
    print("constant scores        :", concordance_index(t, [1.0] * 4, e))  # 0.5
    try:
        from lifelines.utils import concordance_index as li
        print("lifelines agrees      :",
              abs(li(t, [-s for s in perfect], e) - 1.0) < 1e-9)
    except ImportError:
        print("(lifelines not installed; skipping cross-check)")
