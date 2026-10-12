"""The Fed-Heart-Disease baseline model, loss and metric, following FLamby exactly.

model  (fed_heart_disease/model.py)  : Linear(13, 1) followed by a sigmoid, i.e. plain
                                       logistic regression.  14 parameters.
loss   (fed_heart_disease/loss.py)   : binary cross-entropy on the sigmoid output.
metric (fed_heart_disease/metric.py) : accuracy, ``((y_pred > 0.5) == y_true).mean()``.
                                       FLamby notes in a comment that they moved from
                                       ROC-AUC to accuracy because centre 2
                                       (Switzerland) has almost only positive labels,
                                       which makes AUC undefined on some batches.

This is the smallest model in FLamby, and that is exactly why it matters for CFL:
Eq. 51's cosine similarities live in 14 dimensions, so two unrelated updates are not
automatically near-orthogonal, and the approximation noise gamma that swamped the
signal on Fed-ISIC2019 (4 M parameters) is tiny here.  Of all seven FLamby datasets
this is the one where the paper's assumptions are closest to being satisfied.
"""
import numpy as np
import torch
import torch.nn as nn


class Baseline(nn.Module):
    """FLamby's fed_heart_disease Baseline: logistic regression on 13 features."""

    def __init__(self, input_dim=13, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def build_model(cfg=None):
    return Baseline()


def model_factory(cfg=None):
    return lambda: build_model(cfg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------- loss
_BCE = nn.BCELoss()


def bce_loss(output, target):
    """FLamby BaselineLoss: BCE on the already-sigmoided output."""
    return _BCE(output, target)


# -------------------------------------------------------------------- metric
def accuracy(y_true, y_pred):
    """FLamby metric: ((y_pred > 0.5) == y_true).mean()."""
    y_true = np.asarray(y_true).reshape(-1).astype("uint8")
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(((y_pred > 0.5) == y_true).mean())


@torch.no_grad()
def evaluate(model, data, device):
    """-> {'accuracy', 'loss', 'n', 'positive_rate'} for one client dataset."""
    was = model.training
    model.eval()
    x, y = data.all()
    x, y = x.to(device), y.to(device)
    out = model(x)
    loss = float(bce_loss(out, y))
    acc = accuracy(y.cpu().numpy(), out.cpu().numpy())
    model.train(was)
    return {"accuracy": acc, "loss": loss, "n": int(len(y)),
            "positive_rate": float(y.mean())}


if __name__ == "__main__":
    m = Baseline()
    print("Fed-Heart-Disease Baseline: {} parameters, out {}".format(
        count_parameters(m), tuple(m(torch.randn(4, 13)).shape)))
    print("accuracy sanity:", accuracy([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]), "(expect 1.0)")
    print("accuracy sanity:", accuracy([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8]), "(expect 0.0)")
