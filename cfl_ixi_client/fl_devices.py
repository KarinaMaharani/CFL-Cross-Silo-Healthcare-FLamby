"""Clients, server and the CFL primitives of Algorithm 5, for 9-client Fed-IXI.

Adapted from the authors' reference implementation
(github.com/felisat/clustered-federated-learning, kept verbatim in
upstream_reference/) with the changes the Fed-IXI setting requires:

  * local training is FLamby's ``num_updates`` SGD steps on a never-ending shuffled
    stream, not whole epochs -- and the number of steps is set per round by the
    straggler schedule;
  * Eq. 32 can be evaluated with the paper's data-size weights (|D_i| / |D_c|);
  * the bipartition of Eq. 25 is solved EXACTLY (all 2^(n-1)-1 cuts, n <= ~12 here)
    as well as by the reference complete-linkage heuristic, and the two are compared
    every round so the heuristic can never be blamed for a bad split;
  * BatchNorm running statistics are buffers, not parameters: they never enter
    dtheta (hence never the cosine similarities) and are either kept local
    (FLamby/CFL default) or averaged inside the cluster;
  * dropped stragglers contribute no dtheta but still RECEIVE their cluster's
    aggregate, so cluster members stay in sync.
"""
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from .metrics import dice_loss, one_hot, evaluate

DEVICE = torch.device("cpu")


def resolve_device(name="auto"):
    global DEVICE
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable "
                           "(torch {})".format(torch.__version__))
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
    """theta_i <- theta_i + sum_j w_j dtheta_j   (Alg. 5 line 13; w uniform in the paper).

    ``targets`` may be larger than ``sources``: dropped stragglers receive the
    aggregate without having contributed to it.
    """
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
    """(alpha_intra_min, alpha_cross_max, gap) for a labelling of S's rows.

    NaN where undefined.  A 1-vs-1 cut has no intra pair but a well-defined cross
    similarity, so only alpha_intra_min and the gap are NaN there.
    """
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
    if alpha_cross_max != alpha_cross_max:          # NaN
        return float("nan")
    return float(np.sqrt(max(0.0, (1.0 - alpha_cross_max) / 2.0)))


def optimal_bipartition(S):
    """Exact arg min over bipartitions of the max cross similarity (Eq. 25).

    Returns (idx_a, idx_b, alpha_cross_max).  Exhaustive: 2^(n-1)-1 cuts (255 for a
    9-client cluster), which is cheaper than one training step.
    """
    n = len(S)
    best = (None, None, np.inf)
    for mask in range(1, 2 ** (n - 1)):            # client n-1 is always in group b
        a = np.array([i for i in range(n) if mask >> i & 1])
        b = np.array([i for i in range(n) if not mask >> i & 1])
        cm = S[np.ix_(a, b)].max()
        if cm < best[2] - 1e-12:
            best = (a, b, float(cm))
    return best


def complete_linkage_bipartition(S):
    """The reference implementation's greedy solution of the same arg min."""
    labels = AgglomerativeClustering(metric="precomputed", linkage="complete",
                                     n_clusters=2).fit(-S).labels_
    return np.argwhere(labels == 0).flatten(), np.argwhere(labels == 1).flatten()


# ------------------------------------------------------------------- devices
class Client:
    def __init__(self, model_fn, cfg, train_data, test_data, idnum):
        self.model = model_fn().to(DEVICE)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr,
                                         momentum=cfg.momentum)
        self.train_data, self.test_data = train_data, test_data
        self.id, self.name, self.cfg = idnum, train_data.name, cfg
        self.n_train = len(train_data)
        self.W = {k: v for k, v in self.model.named_parameters()}
        self.dW = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        self.W_old = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        self.B = {k: v for k, v in self.model.named_buffers() if v.is_floating_point()}
        self.steps_done = 0

    def synchronize_with_server(self, server):
        copy_(target=self.W, source=server.W)
        copy_(target=self.B, source=server.B)

    def compute_weight_update(self, num_updates):
        """``num_updates`` local SGD steps; dtheta_i = theta_new - theta_old (Alg. 5 l.7).

        ``num_updates == 0`` means this client was dropped by the straggler deadline:
        its dtheta is zeroed and it reports no loss.
        """
        copy_(target=self.W_old, source=self.W)
        self.steps_done = int(num_updates)
        if num_updates <= 0:
            zero_(self.dW)
            return float("nan")
        self.optimizer.param_groups[0]["lr"] *= self.cfg.lr_decay
        self.model.train()
        running = 0.0
        for _ in range(num_updates):
            x, m = self.train_data.next_batch()
            x, m = x.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss = dice_loss(self.model(x), one_hot(m))
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            running += loss.item()
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        if not torch.isfinite(flatten(self.dW)).all():
            raise FloatingPointError("client {} produced a non-finite update".format(self.id))
        return running / num_updates

    def reset(self):
        """Undo the local step: aggregation re-applies the (cluster) mean update."""
        copy_(target=self.W, source=self.W_old)

    def evaluate(self, data=None):
        return evaluate(self.model, data or self.test_data, DEVICE)


class Server:
    def __init__(self, model_fn):
        self.model = model_fn().to(DEVICE)
        self.W = {k: v for k, v in self.model.named_parameters()}
        self.B = {k: v for k, v in self.model.named_buffers() if v.is_floating_point()}

    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([c.dW for c in clients])

    def aggregate_clusterwise(self, client_clusters, weighted=True, bn_stats="local",
                              participating=None):
        """Average inside each cluster and hand the result to every member.

        ``participating`` is a boolean mask over ALL clients; dropped clients are
        excluded from the average but still receive it.
        """
        for cluster in client_clusters:
            contrib = [c for c in cluster
                       if participating is None or participating[c.id]]
            if not contrib:
                continue
            w = [c.n_train for c in contrib] if weighted else None
            reduce_add_average(targets=[c.W for c in cluster],
                               sources=[c.dW for c in contrib], weights=w)
            if bn_stats == "average" and len(contrib) > 1:
                ww = np.array(w if w else [1.0] * len(contrib), dtype=float)
                ww = ww / ww.sum()
                for name in contrib[0].B:
                    avg = sum(float(ww[i]) * c.B[name].data for i, c in enumerate(contrib))
                    for c in cluster:
                        c.B[name].data.copy_(avg)

    def compute_max_update_norm(self, cluster):
        """max_i ||dtheta_i||   (Eq. 33)."""
        if not cluster:
            return float("nan")
        return float(np.max([torch.norm(flatten(c.dW)).item() for c in cluster]))

    def compute_mean_update_norm(self, cluster, weighted=True):
        """|| sum_i (|D_i|/|D_c|) dtheta_i ||   (Eq. 32); plain mean if weighted=False."""
        if not cluster:
            return float("nan")
        stacked = torch.stack([flatten(c.dW) for c in cluster])
        if not weighted:
            return float(torch.norm(stacked.mean(0)).item())
        w = torch.tensor([c.n_train for c in cluster], dtype=stacked.dtype,
                         device=stacked.device)
        return float(torch.norm((stacked * (w / w.sum())[:, None]).sum(0)).item())
