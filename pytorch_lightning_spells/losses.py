from typing import Callable

import torch.nn as nn
import torch.nn.functional as F


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets, weight=None):
        """Cross Entropy with Label Smoothing

        Reference: `wangleiofficial/lable-smoothing-pytorch <https://github.com/wangleiofficial/label-smoothing-pytorch>`_
        """
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        if weight is None:
            loss = -log_preds.sum(dim=-1).mean()
        else:
            loss = -(log_preds.sum(dim=-1) * weight.unsqueeze(0)).mean()
        nll = F.nll_loss(log_preds, targets, weight=weight)
        return linear_combination(loss / n, nll, self.eps)


class MixupSoftmaxLoss(nn.Module):
    """A softmax loss that supports MixUp augmentation.

    Works best with pytorch_lightning_spells.callbacks.MixupCallback

    Reference: `Fast.ai's implementation <https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L6>`_
    """

    def __init__(self, class_weights=None, reduction='mean', label_smooth_eps: float = 0):
        super().__init__()
        # setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction
        self.weight = class_weights
        # self.label_smooth_eps = label_smooth_eps
        if label_smooth_eps:
            self.loss_fn: Callable = LabelSmoothCrossEntropy(eps=label_smooth_eps)
        else:
            self.loss_fn = F.cross_entropy

    def forward(self, output, target):
        weight = self.weight
        if weight is not None:
            weight = self.weight.to(output.device)
        if len(target.size()) == 2:
            loss1 = self.loss_fn(output, target[:, 0].long(), weight=weight)
            loss2 = self.loss_fn(output, target[:, 1].long(), weight=weight)
            assert target.size(1) in (3, 4)
            if target.size(1) == 3:
                lambda_ = target[:, 2]
                d = (loss1 * lambda_ + loss2 * (1-lambda_)).mean()
            else:
                lamb_1, lamb_2 = target[:, 2], target[:, 3]
                d = (loss1 * lamb_1 + loss2 * lamb_2).mean()
        else:
            # This handles the cases without MixUp for backward compatibility
            d = self.loss_fn(output, target, weight=weight)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d
