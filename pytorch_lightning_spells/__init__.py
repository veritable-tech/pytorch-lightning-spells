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
    """

    def __init__(self, ema_alpha: float = 0.02):
        super().__init__()
        self.train_loss_tracker = utils.EMATracker(ema_alpha)

    def get_progress_bar_dict(self):
        # don't show the experiment version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def _should_log(self, flag):
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if isinstance(flag, list):
                return flag[0]
            return flag
        return False

    def training_step_end(self, outputs):
        """Requires training_step() to return a dictionary with a "loss" entry."""
        loss = outputs["loss"].mean()
        self.train_loss_tracker.update(loss.detach())
        if self._should_log(outputs["log"]):
            self.logger.log_metrics({
                "train_loss": self.train_loss_tracker.value
            }, step=self.global_step)
        return loss

    def validation_step_end(self, outputs):
        """Requires validation_step() to return a dictionary with the following entries:

            1. val_loss
            2. pred
            3. target
        """
        self.log('val_loss', outputs['loss'].mean())
        for name, metric in self.metrics:
            metric(
                outputs['pred'].view(-1).cpu(),
                outputs['target'].view(-1).cpu()
            )
            self.log("val_" + name, metric)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        self.log('test_loss', outputs['loss'].mean())
        for name, metric in self.metrics:
            metric(
                outputs['pred'].view(-1).cpu(),
                outputs['target'].view(-1).cpu()
            )
            self.log("test_" + name, metric)
