"""Clients, server and the CFL primitives of Algorithm 5, for 18-client Fed-TCGA-BRCA.

Same structure as ../../cfl-replicate-flamby/cfl_ixi_client/fl_devices.py (adapted from
github.com/felisat/clustered-federated-learning); only the local objective differs --
here it is the Cox partial likelihood on a 40-parameter linear model.

That small parameter count is the point of this dataset: Eq. 51's cosine similarities
are computed in 40 dimensions rather than 4 million, so two unrelated updates are NOT
automatically near-orthogonal and the approximation noise gamma is small.  This is the
regime the CFL paper assumes.
"""
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from .models import cox_loss, evaluate, model_factory

DEVICE = torch.device("cpu")


def resolve_device(name="auto"):
    global DEVICE
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")
    DEVICE = torch.device(name)
    return DEVICE


# ---------------------------------------------------------------- primitives
def copy_(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def zero_(target):
    for name in target:
        target[name].data.zero_()


def reduce_add_average(targets, sources, weights=None):
    """theta_i <- theta_i + sum_j w_j dtheta_j   (Alg. 5 line 13)."""
    if not sources:
        return
    n = len(sources)
    w = (torch.full((n,), 1.0 / n) if weights is None
         else torch.as_tensor(weights, dtype=torch.float32))
    w = w / w.sum()
    for name in targets[0]:
        stacked = torch.stack([s[name].data for s in sources])
        tmp = (stacked * w.to(stacked.device).view(-1, *([1] * (stacked.dim() - 1)))).sum(0)
        for target in targets:
            target[name].data += tmp


def flatten(source):
    return torch.cat([v.flatten() for v in source.values()])


def pairwise_angles(sources):
    """alpha_ij = <dtheta_i, dtheta_j> / (||dtheta_i|| ||dtheta_j||)   (Eq. 51)."""
    flat = torch.stack([flatten(s) for s in sources]).double().cpu()
    flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-12)
    return (flat @ flat.T).numpy()


def separation_stats(S, groups):
    """(alpha_intra_min, alpha_cross_max, gap) for a labelling of S's rows."""
    groups = np.asarray(groups)
    n = len(groups)
    nan = float("nan")
    if n < 2 or len(np.unique(groups)) < 2:
        return nan, nan, nan
    same = groups[:, None] == groups[None, :]
    off = ~np.eye(n, dtype=bool)
    intra, cross = S[same & off], S[~same]
    if cross.size == 0:
        return nan, nan, nan
    if intra.size == 0:
        return nan, float(cross.max()), nan
    return float(intra.min()), float(cross.max()), float(intra.min() - cross.max())


def gamma_bound(alpha_cross_max):
    """sqrt((1 - alpha_cross_max)/2) -- right-hand side of Alg. 5 line 18."""
    if alpha_cross_max != alpha_cross_max:
        return float("nan")
    return float(np.sqrt(max(0.0, (1.0 - alpha_cross_max) / 2.0)))


def optimal_bipartition(S, max_exhaustive=16):
    """Exact arg min over bipartitions of the max cross similarity (Eq. 25)."""
    n = len(S)
    if n > max_exhaustive:
        a, b = complete_linkage_bipartition(S)
        return a, b, float(S[np.ix_(a, b)].max()), False
    best = (None, None, np.inf)
    for mask in range(1, 2 ** (n - 1)):
        a = np.array([i for i in range(n) if mask >> i & 1])
        b = np.array([i for i in range(n) if not mask >> i & 1])
        cm = S[np.ix_(a, b)].max()
        if cm < best[2] - 1e-12:
            best = (a, b, float(cm))
    return best[0], best[1], best[2], True


def complete_linkage_bipartition(S):
    labels = AgglomerativeClustering(metric="precomputed", linkage="complete",
                                     n_clusters=2).fit(-S).labels_
    return np.argwhere(labels == 0).flatten(), np.argwhere(labels == 1).flatten()


# ------------------------------------------------------------------- devices
class Client:
    def __init__(self, cfg, train_data, test_data, idnum):
        self.cfg, self.id, self.name = cfg, idnum, train_data.name
        self.model = model_factory(cfg)().to(DEVICE)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr,
                                         momentum=cfg.momentum)
        self.train_data, self.test_data = train_data, test_data
        self.n_train = len(train_data)
        self.W = {k: v for k, v in self.model.named_parameters()}
        self.dW = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.W_old = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.B = {k: v for k, v in self.model.named_buffers() if v.is_floating_point()}
        self.steps_done = 0

    def synchronize_with_server(self, server):
        copy_(target=self.W, source=server.W)
        if self.B:
            copy_(target=self.B, source=server.B)

    def compute_weight_update(self, num_updates):
        """``num_updates`` local SGD steps; 0 means dropped by the straggler deadline."""
        copy_(target=self.W_old, source=self.W)
        self.steps_done = int(num_updates)
        if num_updates <= 0:
            zero_(self.dW)
            return float("nan")
        self.optimizer.param_groups[0]["lr"] *= self.cfg.lr_decay
        self.model.train()
        running, used = 0.0, 0
        for _ in range(num_updates):
            x, y = self.train_data.next_batch()
            x, y = x.to(DEVICE), y.to(DEVICE)
            if len(y) < 2 or float(y[:, 0].sum()) == 0:
                continue        # no comparable pair -> the Cox gradient is zero
            self.optimizer.zero_grad(set_to_none=True)
            loss = cox_loss(self.model(x), y)
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            running += float(loss)
            used += 1
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        if not torch.isfinite(flatten(self.dW)).all():
            raise FloatingPointError("client {} produced a non-finite update".format(self.id))
        return running / used if used else float("nan")

    def reset(self):
        copy_(target=self.W, source=self.W_old)

    def evaluate(self, data=None):
        return evaluate(self.model, data or self.test_data, DEVICE)


class Server:
    def __init__(self, cfg):
        self.model = model_factory(cfg)().to(DEVICE)
        self.W = {k: v for k, v in self.model.named_parameters()}
        self.B = {k: v for k, v in self.model.named_buffers() if v.is_floating_point()}

    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([c.dW for c in clients])

    def aggregate_clusterwise(self, client_clusters, weighted=True, bn_stats="local",
                              participating=None):
        for cluster in client_clusters:
            contrib = [c for c in cluster if participating is None or participating[c.id]]
            if not contrib:
                continue
            w = [c.n_train for c in contrib] if weighted else None
            reduce_add_average(targets=[c.W for c in cluster],
                               sources=[c.dW for c in contrib], weights=w)

    def compute_max_update_norm(self, cluster):
        if not cluster:
            return float("nan")
        return float(np.max([torch.norm(flatten(c.dW)).item() for c in cluster]))

    def compute_mean_update_norm(self, cluster, weighted=True):
        if not cluster:
            return float("nan")
        stacked = torch.stack([flatten(c.dW) for c in cluster])
        if not weighted:
            return float(torch.norm(stacked.mean(0)).item())
        w = torch.tensor([c.n_train for c in cluster], dtype=stacked.dtype,
                         device=stacked.device)
        return float(torch.norm((stacked * (w / w.sum())[:, None]).sum(0)).item())
