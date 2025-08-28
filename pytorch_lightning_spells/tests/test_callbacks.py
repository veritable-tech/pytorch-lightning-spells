# This is a new file for testing callbacks.
# I will add tests here in the next steps.
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning_spells.callbacks import (
    CutMixCallback, LookaheadCallback, LookaheadModelCheckpoint, MixUpCallback,
    RandomAugmentationChoiceCallback, SnapMixCallback, TelegramCallback)


class MockModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def extract_features(self, x):
        return self.conv(x)

    def get_fc(self):
        return self.fc

    def forward(self, x):
        x = self.extract_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RandomAugmentationChoiceCallbackTest(unittest.TestCase):
    def setUp(self):
        self.callback1 = MagicMock(spec=Callback)
        self.callback2 = MagicMock(spec=Callback)
        self.callbacks = [self.callback1, self.callback2]
        self.trainer = MagicMock(spec=Trainer)
        self.module = MagicMock()
        self.batch = MagicMock()
        self.batch_idx = 0

    def test_on_train_batch_start(self):
        # Test that one of the callbacks is called
        callback = RandomAugmentationChoiceCallback(
            self.callbacks, p=[0.5, 0.5])
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, self.batch_idx)
        c1_called = self.callback1.on_train_batch_start.called
        c2_called = self.callback2.on_train_batch_start.called
        # Either one of them is called, but not both
        self.assertTrue(c1_called or c2_called)
        self.assertFalse(c1_called and c2_called)

    def test_no_op_warmup(self):
        # Test that no callback is called during warmup
        callback = RandomAugmentationChoiceCallback(
            self.callbacks, p=[0.5, 0.5], no_op_warmup=2)
        # First step
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, self.batch_idx)
        self.assertFalse(self.callback1.on_train_batch_start.called)
        self.assertFalse(self.callback2.on_train_batch_start.called)
        # Second step
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, self.batch_idx)
        self.assertFalse(self.callback1.on_train_batch_start.called)
        self.assertFalse(self.callback2.on_train_batch_start.called)
        # Third step
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, self.batch_idx)
        c1_called = self.callback1.on_train_batch_start.called
        c2_called = self.callback2.on_train_batch_start.called
        self.assertTrue(c1_called or c2_called)

    def test_no_op_prob(self):
        # Test that no_op_prob works as expected
        with patch('numpy.random.random', return_value=0.05):
            callback = RandomAugmentationChoiceCallback(
                self.callbacks, p=[0.5, 0.5], no_op_prob=0.1)
            callback.on_train_batch_start(
                self.trainer, self.module, self.batch, self.batch_idx)
            self.assertFalse(self.callback1.on_train_batch_start.called)
            self.assertFalse(self.callback2.on_train_batch_start.called)
        with patch('numpy.random.random', return_value=0.15):
            callback = RandomAugmentationChoiceCallback(
                self.callbacks, p=[0.5, 0.5], no_op_prob=0.1)
            callback.on_train_batch_start(
                self.trainer, self.module, self.batch, self.batch_idx)
            c1_called = self.callback1.on_train_batch_start.called
            c2_called = self.callback2.on_train_batch_start.called
            self.assertTrue(c1_called or c2_called)


@patch('telegram.Bot')
class TelegramCallbackTest(unittest.TestCase):
    def setUp(self):
        self.token = 'test_token'
        self.chat_id = 12345
        self.name = 'test_model'
        self.trainer = MagicMock(spec=Trainer)
        self.module = MagicMock()
        self.trainer.logger_connector = MagicMock()

    def test_on_train_start(self, mock_bot):
        callback = TelegramCallback(
            self.token, self.chat_id, self.name)
        callback.on_train_start(self.trainer, self.module)
        mock_bot.return_value.send_message.assert_called()
        call_args = mock_bot.return_value.send_message.call_args[1]
        self.assertIn('has started training', call_args['text'])

    def test_on_train_end(self, mock_bot):
        callback = TelegramCallback(
            self.token, self.chat_id, self.name)
        callback.on_train_end(self.trainer, self.module)
        mock_bot.return_value.send_message.assert_called()
        call_args = mock_bot.return_value.send_message.call_args[1]
        self.assertIn('has finished training', call_args['text'])

    def test_on_validation_end(self, mock_bot):
        callback = TelegramCallback(
            self.token, self.chat_id, self.name, report_evals=True)
        self.trainer.logger_connector.logged_metrics = {'val_loss': 0.5}
        self.trainer.global_step = 100
        self.trainer.current_epoch = 1
        callback.on_validation_end(self.trainer, self.module)
        mock_bot.return_value.send_message.assert_called()
        call_args = mock_bot.return_value.send_message.call_args[1]
        self.assertIn('Metrics from', call_args['text'])
        self.assertIn('val_loss', call_args['text'])

    def test_report_evals_false(self, mock_bot):
        callback = TelegramCallback(
            self.token, self.chat_id, self.name, report_evals=False)
        self.trainer.logger_connector.logged_metrics = {'val_loss': 0.5}
        callback.on_validation_end(self.trainer, self.module)
        mock_bot.return_value.send_message.assert_not_called()


class LookaheadCallbackTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = MagicMock()
        self.optimizer._backup_and_load_cache = MagicMock()
        self.optimizer._clear_and_load_backup = MagicMock()
        self.trainer = MagicMock(spec=Trainer)
        self.trainer.optimizers = [self.optimizer]
        self.module = MagicMock()

    def test_on_validation_start(self):
        callback = LookaheadCallback()
        callback.on_validation_start(self.trainer, self.module)
        self.optimizer._backup_and_load_cache.assert_called_once()

    def test_on_validation_end(self):
        callback = LookaheadCallback()
        callback.on_validation_end(self.trainer, self.module)
        self.optimizer._clear_and_load_backup.assert_called_once()


class LookaheadModelCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = MagicMock()
        self.optimizer._backup_and_load_cache = MagicMock()
        self.optimizer._clear_and_load_backup = MagicMock()
        self.trainer = MagicMock(spec=Trainer)
        self.trainer.optimizers = [self.optimizer]
        self.module = MagicMock()

    @patch('pytorch_lightning.callbacks.ModelCheckpoint.on_validation_start')
    @patch('pytorch_lightning.callbacks.ModelCheckpoint.on_validation_end')
    def test_lookahead_functionality(self, mock_mc_on_validation_end, mock_mc_on_validation_start):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = LookaheadModelCheckpoint(dirpath=tmpdir)
            callback.on_validation_start(self.trainer, self.module)
            self.optimizer._backup_and_load_cache.assert_called_once()
            mock_mc_on_validation_start.assert_called_once()

            callback.on_validation_end(self.trainer, self.module)
            self.optimizer._clear_and_load_backup.assert_called_once()
            mock_mc_on_validation_end.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CutMix/MixUp tests require a GPU")
class AugmentationCallbackTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_classes = 10
        self.input_size = (3, 32, 32)
        self.inputs = torch.randn(
            self.batch_size, *self.input_size).cuda()
        self.targets = torch.randint(
            0, self.num_classes, (self.batch_size,)).cuda()
        self.batch = [self.inputs, self.targets]
        self.trainer = MagicMock(spec=Trainer)
        self.module = MagicMock()

    def test_mixup_callback_softmax(self):
        callback = MixUpCallback(alpha=0.4, softmax_target=True)
        original_batch = self.batch[0].clone()
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, (self.batch_size, 3))
        self.assertFalse(torch.allclose(new_batch, original_batch))

    def test_mixup_callback_no_softmax(self):
        callback = MixUpCallback(alpha=0.4, softmax_target=False)
        original_batch = self.batch[0].clone()
        original_targets = self.batch[1].clone()
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, original_targets.shape)
        self.assertFalse(torch.allclose(new_batch, original_batch))

    def test_snapmix_callback(self):
        model = MockModel().cuda()
        callback = SnapMixCallback(model, image_size=32, alpha=5.0)
        original_batch = self.batch[0].clone()
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, (self.batch_size, 4))
        self.assertFalse(torch.allclose(new_batch, original_batch))

    def test_cutmix_callback_softmax(self):
        callback = CutMixCallback(alpha=1.0, softmax_target=True)
        original_batch = self.batch[0].clone()
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, (self.batch_size, 3))
        self.assertFalse(torch.allclose(new_batch, original_batch))

    def test_cutmix_callback_no_softmax(self):
        callback = CutMixCallback(alpha=1.0, softmax_target=False)
        original_batch = self.batch[0].clone()
        original_targets = self.batch[1].clone()
        callback.on_train_batch_start(
            self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, original_targets.shape)
        self.assertFalse(torch.allclose(new_batch, original_batch))
