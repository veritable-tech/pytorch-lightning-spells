import socket
from copy import deepcopy
from datetime import datetime
from typing import Optional, Sequence, Tuple

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
# from pytorch_lightning.utilities.exceptions import MisconfigurationException

from .cutmix_utils import cutmix_bbox_and_lam


class RandomAugmentationChoiceCallback(Callback):
    def __init__(self, callbacks: Sequence[Callback], p: Sequence[Callback]):
        self.p = np.asarray(p) / np.sum(p)
        self.callbacks = callbacks
        assert len(p) == len(callbacks)

    def get_callback(self):
        return np.random.choice(self.callbacks, p=self.p)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        return self.get_callback().on_train_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx)


class CutMixCallback(Callback):
    """Assumes the first dimension is batch.

    Reference: https://github.com/rwightman/pytorch-image-models/blob/8c9814e3f500e8b37aae86dd4db10aba2c295bd2/timm/data/mixup.py
    """

    def __init__(self, alpha: float = 0.4, softmax_target: bool = False, minmax: Optional[Tuple[float, float]] = None):
        super().__init__()
        self.alpha = alpha
        self.softmax_target = softmax_target
        self.minmax = minmax

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        old_batch = batch
        batch, targets = batch
        batch_flipped = batch.flip(0).clone()
        lambd = np.random.beta(self.alpha, self.alpha, batch.size(0))
        for i in range(batch.shape[0]):
            (yl, yh, xl, xh), lambd_tmp = cutmix_bbox_and_lam(
                batch.shape, lambd[i], ratio_minmax=self.minmax, correct_lam=True)
            lambd[i] = lambd_tmp
            # fill in the cut regions
            batch[i, :, yl:yh, xl:xh] = batch_flipped[i, :, yl:yh, xl:xh]
        # Create the tensor and expand (for target)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(targets.size())-1)]
        ).expand(-1, *targets.shape[1:])
        # Combine targets
        if self.softmax_target:
            new_targets = torch.stack([
                targets.float(), targets.flip(0).float(), lambd_tensor
            ], dim=1)
        else:
            new_targets = (
                targets * lambd_tensor +
                targets.flip(0) * (1-lambd_tensor)
            )
        old_batch[0] = batch
        old_batch[1] = new_targets


class MixUpCallback(Callback):
    """Callback that perform MixUp augmentation on the input batch.

    Assumes the first dimension is batch.

    Works best with pytorch_lightning_spells.losses.MixupSoftmaxLoss

    Reference: https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py
    """

    def __init__(self, alpha: float = 0.4, softmax_target: bool = False):
        super().__init__()
        self.alpha = alpha
        self.softmax_target = softmax_target

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        old_batch = batch
        batch, targets = batch
        permuted_idx = torch.randperm(batch.size(0)).to(batch.device)
        lambd = np.random.beta(self.alpha, self.alpha, batch.size(0))
        lambd = np.concatenate(
            [lambd[:, np.newaxis], 1-lambd[:, np.newaxis]], axis=1
        ).max(axis=1)
        # Create the tensor and expand (for batch inputs)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(batch.size())-1)]
        ).expand(-1, *batch.shape[1:])
        # Combine input batch
        new_batch = (batch * lambd_tensor +
                     batch[permuted_idx] * (1-lambd_tensor))
        # Create the tensor and expand (for target)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(targets.size())-1)]
        ).expand(-1, *targets.shape[1:])
        # Combine targets
        if self.softmax_target:
            new_targets = torch.stack([
                targets.float(), targets[permuted_idx].float(), lambd_tensor
            ], dim=1)
        else:
            new_targets = (
                targets * lambd_tensor +
                targets[permuted_idx] * (1-lambd_tensor)
            )
        old_batch[0] = new_batch
        old_batch[1] = new_targets


class TelegramCallback(Callback):
    """A Telegram notification callback

    Reference: https://github.com/huggingface/knockknock
    """
    DATE_FORMAT = "%Y-%m-%d %H:%M:%d"

    def __init__(self, token: str, chat_id: int, name: str, report_evals: bool = False):
        try:
            import telegram
        except ImportError:
            raise ImportError(
                "Please install 'python-telegram-bot' before using TelegramCallback.")
        self._token = token
        self.telegram_bot = telegram.Bot(token=self._token)
        self.host_name = socket.gethostname()
        self.report_evals = report_evals
        self.chat_id = chat_id
        self.name = name
        self.start_time = datetime.now()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.start_time = datetime.now()
        contents = [
            f'{self.name} has started training 🎬',
            'Machine name: %s' % self.host_name,
            'Starting date: %s' % self.start_time.strftime(
                TelegramCallback.DATE_FORMAT)
        ]
        text = '\n'.join(contents)
        self.telegram_bot.send_message(chat_id=self.chat_id, text=text)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        end_time = datetime.now()
        elapsed_time = end_time - self.start_time
        contents = [
            f'{self.name} has finished training 🎉',
            'Machine name: %s' % self.host_name,
            'Starting date: %s' % self.start_time.strftime(
                TelegramCallback.DATE_FORMAT),
            'End date: %s' % end_time.strftime(
                TelegramCallback.DATE_FORMAT),
            'Training duration: %s' % str(elapsed_time)
        ]
        text = '\n'.join(contents)
        self.telegram_bot.send_message(chat_id=self.chat_id, text=text)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics, meta = self._collect_metrics(trainer)
        if self.report_evals is False:
            return
        contents = [
            f"Metrics from {self.name} at step {meta['step']} (epoch {meta['epoch']}):"
        ]
        contents += [
            f"{metric_name}: {metric_value:.6f}"
            for metric_name, metric_value in metrics.items()
            if metric_name != "epoch"
        ]
        text = '\n'.join(contents)
        self.telegram_bot.send_message(chat_id=self.chat_id, text=text)

    def _collect_metrics(self, trainer):
        ckpt_name_metrics = deepcopy(trainer.logger_connector.logged_metrics)
        # ckpt_name_metrics.update(trainer.logger_connector.callback_metrics)
        # ckpt_name_metrics.update(trainer.logger_connector.progress_bar_metrics)
        meta = {"step": trainer.global_step, "epoch": trainer.current_epoch}
        return ckpt_name_metrics, meta
