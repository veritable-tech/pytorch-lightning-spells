import pytorch_lightning as pl

from . import callbacks
from . import loggers
from . import losses
from . import optimizers
from . import utils
from . import lr_schedulers
from . import metrics
from . import samplers
from .version import (
    __version__,
    __docs__,
    __author__,
    __author_email__,
    __license__
)


class BaseModule(pl.LightningModule):
    """A boilerplate module with some sensible defaults.

    It logs the exponentially smoothed training losses, and the validation metrics.

    You need to implement the `training_step()` and `validation_step()` methods.
    Please refer to the `training_step_end()` and `validation_step_end()` methods for the expected output format.

    Args:
        ema_alpha (float, optional): the weight of the new training loss for the EMA aggregator. Defaults to 0.02.

    Example:

        >>> class TestModule(BaseModule):
        ...     def __init__(self):
        ...         super().__init__(ema_alpha=0.02)
        ...
        ...     def training_step(self, batch, batch_idx):
        ...         return {
        ...             "loss": torch.tensor(1),
        ...             "log": batch_idx % self.trainer.accumulate_grad_batches == 0
        ...         }
        ...
        ...     def validation_step(self, batch, batch_idx):
        ...         return {
        ...             'loss': torch.tensor(1),
        ...             'preds': torch.ones_like(batch[1]),
        ...             'target': batch[1]
        ...         }
        ...
        >>> module = TestModule()
    """

    def __init__(self, ema_alpha: float = 0.02):
        super().__init__()
        self.train_loss_tracker = utils.EMATracker(ema_alpha)

    def get_progress_bar_dict(self):
        """Removes `v_num` from the progress bar.
        """
        # don't show the experiment version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def _should_log(self, flag):
        """Determines if the loss of this training step should be logged.

        Args:
            flag (Union[bool, List[bool]]): if this is a loggable step.

        Returns:
            bool: True if this step should be logged.
        """
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if isinstance(flag, list):
                # for distributed training scenarios
                return flag[0]
            return flag
        return False

    def training_step_end(self, outputs):
        """This method logs the training loss for you.

        It follows the `log_every_n_steps` attribute of the associated Trainer.

        The output from `.validation_step()` method must contains these two entries:

        1. loss: the training loss.
        2. log: a boolean value indicating if this is a loggable step.


        A loggable step is a step that involves an optimizer step. The opposite is a step that only updates the gradients but not the parameters(e.g., in gradient accumulation).

        Args:
            outputs (Dict): the output from `.training_step()` method.
        """
        loss = outputs["loss"].mean()
        self.train_loss_tracker.update(loss.detach())
        if self._should_log(outputs["log"]):
            self.logger.log_metrics({
                "train_loss": self.train_loss_tracker.value
            }, step=self.global_step)
        return loss

    def validation_step_end(self, outputs):
        """This method logs the validation loss and metrics for you.

        The output from `.validation_step()` method must contains these three entries:

            1. loss: the validation loss.
            2. pred: the predicted labels or values.
            3. target: the ground truth lables or values.

        Args:
            outputs (Dict): the output from `.validation_step()` method.
        """
        self.log('val_loss', outputs['loss'].mean())
        for name, metric in self.metrics:
            metric(
                outputs['pred'].view(-1).cpu(),
                outputs['target'].view(-1).cpu()
            )
            self.log("val_" + name, metric)

    def test_step(self, batch, batch_idx):
        """Simply defer to `.validation_step()` method."""
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        """Basically the same as `.validation_step_ends()` method, but with a different prefix.
        """
        # TODO: refactor to be able to simply defer to `.validation_step_ends()`?
        self.log('test_loss', outputs['loss'].mean())
        for name, metric in self.metrics:
            metric(
                outputs['pred'].view(-1).cpu(),
                outputs['target'].view(-1).cpu()
            )
            self.log("test_" + name, metric)
