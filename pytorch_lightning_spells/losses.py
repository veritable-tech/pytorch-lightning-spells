from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Poly1CrossEntropyLoss(nn.Module):
    """Poly-1 Cross-Entropy Loss

    Adapted from `abhuse/polyloss-pytorch <https://github.com/abhuse/polyloss-pytorch>`_.

    Reference: `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions <http://arxiv.org/abs/2204.12511>`_.
    """
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(
            labels, num_classes=self.num_classes
        ).to(device=logits.device, dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(
            input=logits,
            target=labels,
            reduction='none',
            weight=self.weight
        )
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class Poly1FocalLoss(nn.Module):
    """Poly-1 Focal Loss

    Adapted from `abhuse/polyloss-pytorch <https://github.com/abhuse/polyloss-pytorch>`_.

    Reference: `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions <http://arxiv.org/abs/2204.12511>`_.
    """
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: Tensor = None,
                 label_is_onehot: bool = False):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(
                    1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1


def _linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


class LabelSmoothCrossEntropy(nn.Module):
    """Cross Entropy with Label Smoothing

    Reference: `wangleiofficial/lable-smoothing-pytorch <https://github.com/wangleiofficial/label-smoothing-pytorch>`_

    The ground truth label will have a value of `1-eps` in the target vector.

    Args:
            eps (float): the smoothing factor.
    """

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets, weight=None):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        if weight is None:
            loss = -log_preds.sum(dim=-1).mean()
        else:
            loss = -(log_preds.sum(dim=-1) * weight.unsqueeze(0)).mean()
        nll = F.nll_loss(log_preds, targets, weight=weight)
        return _linear_combination(loss / n, nll, self.eps)


class MixupSoftmaxLoss(nn.Module):
    """A softmax loss that supports MixUp augmentation.

    It requires the input batch to be manipulated into certain format.
    Works best with MixUpCallback, CutMixCallback, and SnapMixCallback.

    Reference: `Fast.ai's implementation <https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L6>`_

    Args:
        class_weights (torch.Tensor, optional): The weight of each class. Defaults to the same weight.
        reduction (str, optional): Loss reduction method. Defaults to 'mean'.
        label_smooth_eps (float, optional): If larger than zero, use `LabelSmoothedCrossEntropy` instead of `CrossEntropy`. Defaults to 0.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None, reduction: str = 'mean', label_smooth_eps: float = 0):
        super().__init__()
        # setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction
        self.weight = class_weights
        # self.label_smooth_eps = label_smooth_eps
        if label_smooth_eps:
            self.loss_fn: Callable = LabelSmoothCrossEntropy(eps=label_smooth_eps)
        else:
            self.loss_fn = F.cross_entropy

    def forward(self, output: torch.Tensor, target):
        """The feed-forward.

        The target tensor should have three columns:

        1. the first class.
        2. the second class.
        3. the lambda value to mix the above two classes.

        Args:
            output (torch.Tensor): the model output.
            target (torch.Tensor): Shaped (batch_size, 3).

        Returns:
            torch.Tensor: the result loss
        """
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
