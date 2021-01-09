import torch.nn as nn
import torch.nn.functional as F


class MixupSoftmaxLoss(nn.Module):
    """A softmax loss that supports MixUp augmentation.

    Works best with pytorch_lightning_spells.callbacks.MixupCallback

    Reference: https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L6
    """

    def __init__(self, class_weights=None, reduction='mean'):
        super().__init__()
        # setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction
        self.weight = class_weights

    def forward(self, output, target):
        weight = self.weight.to(output.device)
        if len(target.size()) == 2:
            loss1 = F.cross_entropy(output, target[:, 0].long(), weight=weight)
            loss2 = F.cross_entropy(output, target[:, 1].long(), weight=weight)
            lambda_ = target[:, 2]
            d = (loss1 * lambda_ + loss2 * (1-lambda_)).mean()
        else:
            # This handles the cases without MixUp for backward compatibility
            d = F.cross_entropy(output, target, weight=weight)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d
