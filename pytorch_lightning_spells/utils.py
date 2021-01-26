import warnings
from typing import Sequence, Union

import torch
import numpy as np

Layer = Union[torch.nn.Module, torch.nn.ModuleList]


class EMATracker:
    """Keeps the exponential moving average for a single series.

    Parameters
    ----------
    alpha : float, optional
        the weight of the new value, by default 0.05
    """

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self._value = None

    def update(self, new_value):
        new_value = new_value.detach()
        if torch.isnan(new_value):
            warnings.warn("NaN encountered as training loss!")
            return
        if self._value is None:
            self._value = new_value
        else:
            self._value = (
                new_value * self.alpha +
                self._value * (1-self.alpha)
            )

    @property
    def value(self):
        return self._value


def count_parameters(parameters):
    return int(np.sum(list(p.numel() for p in parameters)))

# -----------------------------------
# Layer freezing from fast.ai v1
# -----------------------------------


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, torch.nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(layer: Layer, trainable: bool):
    """Freeze or unfreeze all parameters in the layer."""
    apply_leaf(layer, lambda m: set_trainable_attr(m, trainable))


def freeze_layers(layer_groups: Sequence[Layer], freeze_flags: Sequence[bool]):
    """Freeze or unfreeze groups of layers"""
    assert len(freeze_flags) == len(layer_groups)
    for layer, flag in zip(layer_groups, freeze_flags):
        set_trainable(layer, not flag)

# -----------------------------------------------------------
# Separate BatchNorm2d and GroupNorm paremeters from others
# -----------------------------------------------------------


def seperate_parameters(module):
    decay, no_decay = [], []
    if isinstance(module, list):
        for entry in module:
            tmp = seperate_parameters(entry)
            decay.extend(tmp[0])
            no_decay.extend(tmp[1])
    else:
        for submodule in module.modules():
            if not list(submodule.children()):  # leaf node
                if isinstance(submodule, (torch.nn.GroupNorm, torch.nn.BatchNorm2d)):
                    no_decay.extend(list(submodule.parameters()))
                else:
                    decay.extend(list(submodule.parameters()))
    return decay, no_decay
