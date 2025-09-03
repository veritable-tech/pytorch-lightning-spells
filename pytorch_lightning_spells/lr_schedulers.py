import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

__all__ = [
    "BaseLRScheduler",
    "LinearLR",
    "ExponentialLR",
    "MultiStageScheduler",
    "CosineAnnealingScheduler",
]


class BaseLRScheduler(_LRScheduler):
    def switch_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer._step_count = self._step_count

    def clear_optimizer(self):
        self.optimizer = None


class CosineAnnealingScheduler(CosineAnnealingLR, BaseLRScheduler):
    pass


class LinearLR(BaseLRScheduler):
    """Linearly increases or decrease the learning rate between two boundaries over a number of
    iterations.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr_ratio: float,
        total_epochs: float,
        upward: bool = True,
        last_epoch: int = -1,
    ):
        """Initialize a scheduler.

        Args:
            optimizer (Union[torch.optim.Optimizer, apex.fp16_utils.fp16_optimizer.FP16_Optimizer]):
            min_lr_ratio:  min_lr_ratio * base_lr will be the starting learning rate.
            total_epochs: the total number of "steps" in this run.
            upward: whether the learning rate goes up or down. Defaults to True.
            last_epoch: the index of last epoch. Defaults to -1.
        """
        assert min_lr_ratio < 1
        self.upward = upward
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs - 1  # starts from zero
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch
        if self.upward:
            progress = 1 - current_epoch / self.total_epochs  # 1 to 0
        else:
            progress = current_epoch / self.total_epochs  # 0 to 1
        # safety measure
        progress = max(min(progress, 1.0), 0.0)
        return [base_lr - progress * (base_lr - self.min_lr_ratio * base_lr) for base_lr in self.base_lrs]


class ExponentialLR(BaseLRScheduler):
    """Exponentially increases the learning rate between two boundaries over
    a number of iterations.

    Mainly used by LR finders.
    """

    def __init__(self, optimizer, min_lr_ratio, total_epochs, last_epoch=-1):
        """Initialize a scheduler.

        Parameters
        ----------
        optimizer : Union[torch.optim.Optimizer, apex.fp16_utils.fp16_optimizer.FP16_Optimizer]
        min_lr_ratio : float
            min_lr_ratio * base_lr will be the starting learning rate.
        total_epochs : int
            the total number of "steps" in this run.
        last_epoch : int, optional
            the index of last epoch, by default -1.
        """
        assert min_lr_ratio < 1
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs - 1  # start from zero
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        progress = 1 - current_epoch / self.total_epochs  # 1 to 0
        return [base_lr * (self.min_lr_ratio) ** progress for base_lr in self.base_lrs]


class MultiStageScheduler(_LRScheduler):
    def __init__(
        self,
        schedulers: list[_LRScheduler] | tuple[_LRScheduler],
        start_at_epochs: list[int] | tuple[int],
        last_epoch: int = -1,
    ):
        assert len(schedulers) == len(start_at_epochs)
        # sort starting epochs in descending order
        idx = np.flip(np.argsort(np.asarray(start_at_epochs)))
        self.schedulers = np.asarray(schedulers)[idx]
        self.start_at_epochs = np.asarray(start_at_epochs)[idx]
        self.last_epoch = last_epoch
        # Explicitly run step(). Otherwise the initial LR will be initialized by the last sub-scheduler
        self.step(0)
        self.optimizer = self.schedulers[0].optimizer

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch - 1
        for scheduler, starting_epoch in zip(self.schedulers, self.start_at_epochs):
            if self.last_epoch + 1 >= starting_epoch:
                scheduler.last_epoch = self.last_epoch - starting_epoch
                return scheduler.step()

    def switch_optimizer(self, optimizer):
        for scheduler in self.schedulers:
            scheduler.optimizer = optimizer
            scheduler.optimizer._step_count = scheduler._step_count
        self.optimizer = self.schedulers[0].optimizer

    def clear_optimizer(self):
        for scheduler in self.schedulers:
            scheduler.optimizer = None
        self.optimizer = None

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        results = {key: value for key, value in self.__dict__.items() if key != "optimizer"}
        del results["schedulers"]
        for i, scheduler in enumerate(self.schedulers):
            results["schedulers_" + str(i)] = scheduler.state_dict()
        return results

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        for i, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(state_dict["schedulers_" + str(i)])
            del state_dict["schedulers_" + str(i)]
        self.__dict__.update(state_dict)
        # Manually bind optimizer to make sure
        self.switch_optimizer(self.optimizer)
