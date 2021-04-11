import socket
from copy import deepcopy
from datetime import datetime
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.exceptions import MisconfigurationException

from .cutmix_utils import cutmix_bbox_and_lam, rand_bbox, rand_bbox_minmax
from .snapmix_utils import get_spm


class RandomAugmentationChoiceCallback(Callback):
    """Randomly pick an augmentation callback to use for each batch.

    Also supports no-op warmups and no-op probability.

    Args:
        callbacks (Sequence[Callback]): A sequence of calbacks to choose from.
        p (Sequence[Callback]): A sequence of probabilities for the callbacks.
        no_op_warmup (int, optional): the number of initial steps that should not have any augmentation. Defaults to 0.
        no_op_prob (float, optional): the probability of a step that has no augmentation. Defaults to 0.
    """
    def __init__(
            self, callbacks: Sequence[Callback], p: Sequence[Callback],
            no_op_warmup: int = 0, no_op_prob: float = 0):
        self.p = np.asarray(p) / np.sum(p)
        self.callbacks = callbacks
        self.no_op_warmup = no_op_warmup
        self.step = 0
        self.no_op_prob = no_op_prob
        assert len(p) == len(callbacks)

    def get_callback(self):
        return np.random.choice(self.callbacks, p=self.p)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.step += 1
        if self.no_op_warmup >= self.step:
            return
        if self.no_op_prob and np.random.random() < self.no_op_prob:
            return
        self.get_callback().on_train_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx)


class SnapMixCallback(Callback):
    """Callback that perform SnapMix augmentation on the input batch.

    Reference: `Shaoli-Huang/SnapMix <https://github.com/Shaoli-Huang/SnapMix/>`_

    Warning:
        1. **Requires the model to have implemented `extract_features` and `get_fc` methods.**
        2. Can only run in CUDA-enabled environments.
    """

    def __init__(
            self, model, image_size, half: bool = False,
            minmax: Optional[Tuple[float, float]] = None,
            cutmix_bbox: bool = False,  # cutmix style randbox
            alpha: float = 0.4, softmax_target: bool = True):
        self._model = model
        self._half = half
        self.image_size = image_size
        self.alpha = alpha
        self.minmax = minmax
        self.cutmix_bbox = cutmix_bbox
        assert softmax_target, "SnapMix only support softmax_target=True"

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        old_batch = batch
        batch, targets = batch
        target_activation_map = get_spm(
            batch, targets, self._model, self.image_size, self._half
        )
        bs = batch.size(0)
        lamb_1 = np.clip(np.random.beta(self.alpha, self.alpha), 0.05, 0.95)
        lamb_2 = np.clip(np.random.beta(self.alpha, self.alpha), 0.05, 0.95)
        rand_index = torch.randperm(bs).cuda()
        target_activation_map_b = target_activation_map[rand_index, :, :]

        # same_label = target == target_b

        area_1, area_2, cnt = 0, 0, 0
        while area_1 <= 0 or area_2 <= 0:
            if self.minmax:
                bby1_1, bby2_1, bbx1_1, bbx2_1 = rand_bbox_minmax(
                    batch.size(), self.minmax)
                bby1_2, bby2_2, bbx1_2, bbx2_2 = rand_bbox_minmax(
                    batch.size(), self.minmax)
            else:
                bby1_1, bby2_1, bbx1_1, bbx2_1 = rand_bbox(batch.size(), lamb_1)
                bby1_2, bby2_2, bbx1_2, bbx2_2 = rand_bbox(batch.size(), lamb_2)
            if self.cutmix_bbox is True:
                # Use only one random bounding box
                bby1_2, bby2_2, bbx1_2, bbx2_2 = bby1_1, bby2_1, bbx1_1, bbx2_1
            area_1 = (bby2_1-bby1_1) * (bbx2_1-bbx1_1)
            area_2 = (bby2_2-bby1_2) * (bbx2_2-bbx1_2)
            cnt += 1
            # Avoid infinite loops when something goes wrong
            assert cnt < 10, f"{lamb_1}, {lamb_2}"

        cropped = batch[rand_index, :, bby1_2:bby2_2, bbx1_2:bbx2_2].clone()
        if self.cutmix_bbox is False:
            cropped = F.interpolate(
                cropped,
                size=(bby2_1-bby1_1, bbx2_1-bbx1_1),
                mode='bilinear',
                align_corners=True
            )
        batch[:, :, bby1_1:bby2_1, bbx1_1:bbx2_1] = cropped
        lamb_1 = (
            1 - target_activation_map[
                :, bby1_1:bby2_1, bbx1_1:bbx2_1
            ].sum(1).sum(1)
        )
        lamb_2 = target_activation_map_b[
            :, bby1_2:bby2_2, bbx1_2:bbx2_2
        ].sum(1).sum(1)

        # Fall back to Cutmix lambda
        lamb_cutmix = 1 - (
            (bbx2_1 - bbx1_1) * (bby2_1 - bby1_1) /
            (batch.size(2) * batch.size(3))
        )
        lamb_1[torch.isnan(lamb_1)] = lamb_cutmix
        lamb_2[torch.isnan(lamb_2)] = 1 - lamb_cutmix
        # Combine targets
        new_targets = torch.stack([
            targets.float(), targets[rand_index].float(),
            lamb_1.to(targets.device), lamb_2.to(targets.device)
        ], dim=1)
        old_batch[0] = batch
        old_batch[1] = new_targets


class CutMixCallback(Callback):
    """Callback that perform CutMix augmentation on the input batch.

    Assumes the first dimension is batch.

    Reference: `rwightman/pytorch-image-models/ <https://github.com/rwightman/pytorch-image-models/blob/8c9814e3f500e8b37aae86dd4db10aba2c295bd2/timm/data/mixup.py>`_
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

    Reference: `Fast.ai's implementation <https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py>`_
    """

    def __init__(self, alpha: float = 0.4, softmax_target: bool = False):
        super().__init__()
        self.alpha = alpha
        self.softmax_target = softmax_target

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        old_batch = batch
        batch, targets = batch
        batch_flipped = batch.flip(0).clone()
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
                     batch_flipped * (1-lambd_tensor))
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
        old_batch[0] = new_batch
        old_batch[1] = new_targets


class TelegramCallback(Callback):
    """A Telegram notification callback

    Reference: `huggingface/knockknock <https://github.com/huggingface/knockknock>`_
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

    def send_message(self, text):
        try:
            self.telegram_bot.send_message(chat_id=self.chat_id, text=text)
        except telegram.error.TimeOut:
            # Ignore timeouts and continue training
            pass

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.start_time = datetime.now()
        contents = [
            f'{self.name} has started training 🎬',
            'Machine name: %s' % self.host_name,
            'Starting date: %s' % self.start_time.strftime(
                TelegramCallback.DATE_FORMAT)
        ]
        text = '\n'.join(contents)
        self.send_message(text=text)

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
        self.send_message(text=text)

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
        self.send_message(text=text)

    def _collect_metrics(self, trainer):
        ckpt_name_metrics = deepcopy(trainer.logger_connector.logged_metrics)
        # ckpt_name_metrics.update(trainer.logger_connector.callback_metrics)
        # ckpt_name_metrics.update(trainer.logger_connector.progress_bar_metrics)
        meta = {"step": trainer.global_step, "epoch": trainer.current_epoch}
        return ckpt_name_metrics, meta


class LookaheadCallback(Callback):
    """Switch to the slow weights before evaluation and switch back after.
    """

    def on_validation_start(self, trainer, pl_module):
        optimizer = trainer.optimizers(use_pl_optimizer=False)
        if hasattr(optimizer, "_backup_and_load_cache"):
            print("load slow parameters")
            optimizer._backup_and_load_cache()

    def on_validation_end(self, trainer, pl_module):
        optimizer = trainer.optimizers(use_pl_optimizer=False)
        if hasattr(optimizer, "_clear_and_load_backup"):
            print("load fast parameters")
            optimizer._clear_and_load_backup()


class LookaheadModelCheckpoint(ModelCheckpoint):
    """Combines LookaheadCallback and ModelCheckpoint
    """

    def on_validation_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            if hasattr(optimizer, "_backup_and_load_cache"):
                print("load slow parameters")
                optimizer._backup_and_load_cache()

    def on_validation_end(self, trainer, pl_module):
        self.save_checkpoint(trainer, pl_module)
        for optimizer in trainer.optimizers:
            if hasattr(optimizer, "_clear_and_load_backup"):
                print("load fast parameters")
                optimizer._clear_and_load_backup()
