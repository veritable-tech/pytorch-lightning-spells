import math
import warnings
from typing import Sequence, Union, Iterable, Optional, List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

Layer = Union[torch.nn.Module, torch.nn.ModuleList]


class EMATracker:
    """Keeps the exponential moving average for a single series.

    Args:
        alpha (float, optional): the weight of the new value, by default 0.05

    Examples:

        >>> tracker = EMATracker(0.1)
        >>> tracker.update(1.)
        >>> tracker.value
        1.0
        >>> tracker.update(2.)
        >>> tracker.value # 1 * 0.9 + 2 * 0.1
        1.1
        >>> tracker.update(float('nan')) # this won't have any effect
        >>> tracker.value
        1.1
    """

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self._value: Optional[Union[float, torch.Tensor]] = None

    def update(self, new_value: Union[float, torch.Tensor]):
        """Adds a new value to the tracker.

        It will ignore NaNs and raise a warning in those cases.

        Args:
            new_value (Union[float, torch.Tensor]): the incoming value.
        """
        if isinstance(new_value, torch.Tensor):
            new_value = new_value.detach()
            if torch.isnan(new_value):
                warnings.warn("NaN encountered as training loss!")
                return
        elif math.isnan(new_value):
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
        """The smoothed value.
        """
        return self._value


def count_parameters(parameters: Iterable[Union[torch.Tensor, Parameter]]):
    """Count the number of parameters

    Args:
        parameters (Iterable[Union[torch.Tensor, Parameter]]): parameters you want to count.

    Returns:
        int: the number of parameters counted.

    Example:

        >>> count_parameters([torch.rand(100), torch.rand(10)])
        110
        >>> count_parameters([torch.rand(100, 2), torch.rand(10, 3)])
        230
    """
    return int(np.sum(list(p.numel() for p in parameters)))

# -----------------------------------
# Layer freezing from fast.ai v1
# -----------------------------------


def _children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def _set_trainable_attr(m, b):
    if isinstance(m, torch.nn.Parameter):
        m.requires_grad = b
        return
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def _apply_leaf(m, f):
    if isinstance(m, (torch.nn.Module, torch.nn.Parameter)):
        f(m)
        if isinstance(m, torch.nn.Parameter):
            return
    c = _children(m)
    if len(c) > 0:
        for l in c:
            _apply_leaf(l, f)


def set_trainable(layer: Layer, trainable: bool):
    """Freeze or unfreeze all parameters in the layer.

    Args:
        layer (Union[torch.nn.Module, torch.nn.ModuleList]): the target layer
        trainable (bool): True to unfreeze; False to freeze

    Example:

        >>> model = nn.Sequential(nn.Linear(10, 100), nn.Linear(100, 1))
        >>> model[0].weight.requires_grad
        True
        >>> set_trainable(model, False)
        >>> model[0].weight.requires_grad
        False
        >>> set_trainable(model, True)
        >>> model[0].weight.requires_grad
        True
    """
    _apply_leaf(layer, lambda m: _set_trainable_attr(m, trainable))


def freeze_layers(layer_groups: Sequence[Layer], freeze_flags: Sequence[bool]):
    """Freeze or unfreeze groups of layers

    Args:
        layer_groups (Sequence[Layer]): the target lists of layers
        freeze_flags (Sequence[bool]): the corresponding trainable flags


    .. warning::

        The value in `freeze_flag` has the opposite meaning as in `trainable` of `set_trainable`.

        Set True to freeze; False to unfreeze.

    Examples:

        >>> model = nn.Sequential(nn.Linear(10, 100), nn.Linear(100, 1))
        >>> freeze_layers([model[0], model[1]], [True, False])
        >>> model[0].weight.requires_grad
        False
        >>> model[1].weight.requires_grad
        True
        >>> freeze_layers([model[0], model[1]], [False, True])
        >>> model[0].weight.requires_grad
        True
        >>> model[1].weight.requires_grad
        False
    """
    assert len(freeze_flags) == len(layer_groups)
    for layer, flag in zip(layer_groups, freeze_flags):
        set_trainable(layer, not flag)

# -----------------------------------------------------------
# Separate BatchNorm2d and GroupNorm paremeters from others
# -----------------------------------------------------------


def separate_parameters(module: Union[Parameter, nn.Module, List[nn.Module]]):
    """Separate BatchNorm2d, GroupNorm, and LayerNorm paremeters from others

    Args:
        module (Union[Parameter, nn.Module, List[nn.Module]]): to be separated.

    Returns:
        Tuple[List[Parameter], List[Parameter]]: lists of decay and no-decay parameters.

    Example:

        >>> model = nn.Sequential(nn.Linear(100, 10, bias=False), nn.BatchNorm1d(10))
        >>> _ = nn.init.constant_(model[0].weight, 2.)
        >>> _ = nn.init.constant_(model[1].weight, 1.)
        >>> _ = nn.init.constant_(model[1].bias, 0.)
        >>> model[0].weight.data.sum().item()
        2000.0
        >>> model[1].weight.data.sum().item()
        10.0
        >>> decay, no_decay = separate_parameters(model) # separate the parameters
        >>> np.sum([x.sum().detach().numpy() for x in decay]) # nn.Linear
        2000.0
        >>> np.sum([x.sum().detach().numpy() for x in no_decay]) # nn.BatchNorm1d
        10.0
        >>> optimizer = torch.optim.AdamW([{
        ...    "params": decay, "weight_decay": 0.1
        ... }, {
        ...    "params": no_decay, "weight_decay": 0
        ... }], lr = 1e-3)
    """
    decay, no_decay = [], []
    if isinstance(module, list):
        for entry in module:
            tmp = separate_parameters(entry)
            decay.extend(tmp[0])
            no_decay.extend(tmp[1])
    elif isinstance(module, torch.nn.Parameter):
        no_decay.append(module)
    else:
        for submodule in module.modules():
            if not list(submodule.children()):  # leaf node
                if isinstance(submodule, (
                        torch.nn.GroupNorm, torch.nn.BatchNorm2d,
                        torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
                    no_decay.extend(list(submodule.parameters()))
                elif isinstance(submodule, torch.nn.Identity):
                    continue
                else:
                    decay.extend(list(submodule.parameters()))
    return decay, no_decay
