"""Fed-IXI loss and metrics, following FLamby exactly.

loss   (fed_ixi/loss.py)   : mean over (batch, class) of 1 - 2TP / (2TP + FP + FN + 1e-9)
                             computed on the SOFTMAX output, both channels.
metric (fed_ixi/metric.py) : soft Dice  sum(p*g) / (sum(0.5*(p+g)) + 1e-7), with
                             dice := 1 where the union is empty, averaged over every
                             (sample, class) pair -- background included, NO threshold.
                             This is the number the FLamby paper reports (~0.98).

Because background is ~77 % of the volume, the FLamby metric is compressed near 1.
We additionally log the thresholded foreground Dice (argmax, brain channel only),
which is the clinically usual number and separates models far better.
"""
import torch

SPATIAL = (2, 3, 4)


def one_hot(mask):
    """uint8/bool foreground mask (N, D, H, W) -> float one-hot (N, 2, D, H, W)."""
    fg = mask.float().unsqueeze(1)
    return torch.cat((1.0 - fg, fg), dim=1)


def dice_loss(output, target, eps=1e-9):
    """FLamby BaselineLoss."""
    tp = (output * target).sum(dim=SPATIAL)
    fp = (output * (1 - target)).sum(dim=SPATIAL)
    fn = ((1 - output) * target).sum(dim=SPATIAL)
    return torch.mean(1 - 2 * tp / (2 * tp + fp + fn + eps))


@torch.no_grad()
def soft_dice_flamby(output, target):
    """Per-(sample, class) soft Dice as in FLamby's metric, shape (N, 2)."""
    inter = (output * target).sum(dim=SPATIAL)
    union = (0.5 * (output + target)).sum(dim=SPATIAL)
    dice = inter / (union + 1e-7)
    dice[union == 0] = 1.0
    return dice


@torch.no_grad()
def hard_dice_fg(output, target):
    """Per-sample Dice of the thresholded brain mask, shape (N,)."""
    pred = (output.argmax(1) == 1).float()
    gt = target[:, 1]
    inter = (pred * gt).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = 2 * inter / denom.clamp_min(1e-7)
    dice[denom == 0] = 1.0
    return dice


@torch.no_grad()
def evaluate(model, data, device, batch_size=8):
    """-> {'dice', 'dice_fg', 'loss', 'dice_fg_samples'} for one client dataset."""
    was_training = model.training
    model.eval()
    soft, hard, loss, n = [], [], 0.0, 0
    for x, m in data.batches(shuffle=False, batch_size=batch_size):
        x, t = x.to(device, non_blocking=True), one_hot(m.to(device, non_blocking=True))
        out = model(x)
        soft.append(soft_dice_flamby(out, t).cpu())
        hard.append(hard_dice_fg(out, t).cpu())
        loss += dice_loss(out, t).item() * x.shape[0]
        n += x.shape[0]
    model.train(was_training)
    hard = torch.cat(hard)
    return {"dice": float(torch.cat(soft).mean()),
            "dice_fg": float(hard.mean()),
            "loss": loss / n,
            "dice_fg_samples": hard.tolist()}   # per test subject, for paired CIs
