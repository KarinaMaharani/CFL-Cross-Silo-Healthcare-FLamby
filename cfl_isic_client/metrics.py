"""Fed-ISIC2019 loss and metric, following FLamby exactly.

loss   (fed_isic2019/loss.py)   : weighted focal loss
                                  -(1 - p_t)^gamma * alpha_t * log p_t, averaged,
                                  with gamma = 2 and FLamby's per-class alpha.
metric (fed_isic2019/metric.py) : balanced accuracy -- sklearn's
                                  balanced_accuracy_score(y_true, argmax(logits)),
                                  i.e. the mean per-class recall, which is the right
                                  metric for a dataset where one class is 94 % of a
                                  client's images.

The alpha weights come straight from FLamby and are inversely related to class
frequency: the rarest classes (DF 101x, VASC 93x) dominate the gradient, which is
also why ISIC training is noisy enough to need gradient clipping.
"""
import numpy as np
import torch
import torch.nn.functional as F

#: flamby/datasets/fed_isic2019/loss.py, BaselineLoss.__init__
FOCAL_ALPHA = torch.tensor([5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707,
                            92.5224, 38.3443])
GAMMA = 2.0


def focal_loss(inputs, targets, gamma=GAMMA):
    """FLamby BaselineLoss (weighted focal loss) on raw logits."""
    logpt = F.log_softmax(inputs.float(), dim=1).gather(1, targets.view(-1, 1)).view(-1)
    pt = logpt.exp()
    at = FOCAL_ALPHA.to(inputs.device).gather(0, targets.view(-1))
    return (-1 * (1 - pt) ** gamma * (logpt * at)).mean()


def balanced_accuracy(conf):
    """conf[true, pred] counts -> sklearn balanced_accuracy_score (absent classes ignored)."""
    support = conf.sum(1)
    present = support > 0
    recall = conf.diag()[present].float() / support[present].float()
    return float(recall.mean())


def per_class_recall(conf):
    support = conf.sum(1)
    out = []
    for c in range(conf.shape[0]):
        out.append(float(conf[c, c]) / float(support[c]) if support[c] > 0 else float("nan"))
    return out


@torch.no_grad()
def evaluate(model, data, cfg, device, autocast_ctx):
    """-> {'bacc', 'acc', 'loss', 'conf', 'recall'} for one client dataset."""
    was = model.training
    model.eval()
    conf = torch.zeros(8, 8, dtype=torch.long, device=device)
    loss, n = 0.0, 0
    from .data_utils import test_transform
    for x, y in data.batches(cfg.eval_batch_size):
        x = test_transform(x.to(device, non_blocking=True))
        x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with autocast_ctx():
            out = model(x)
        loss += focal_loss(out, y).item() * len(y)
        n += len(y)
        conf.index_put_((y, out.float().argmax(1)), torch.ones_like(y), accumulate=True)
    model.train(was)
    c = conf.cpu()
    return {"bacc": balanced_accuracy(c),
            "acc": float(c.diag().sum() / c.sum()),
            "loss": loss / max(n, 1),
            "recall": per_class_recall(c),
            "conf": c.tolist()}
