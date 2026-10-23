# This is a new file for testing callbacks.
# I will add tests here in the next steps.
import io
import os
import urllib.error
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

from pytorch_lightning_spells.callbacks import (
    NtfyCallback,
    MixUpCallback,
    CutMixCallback,
    SnapMixCallback,
    TelegramCallback,
    LookaheadCallback,
    LookaheadModelCheckpoint,
    RandomAugmentationChoiceCallback,
)
from pytorch_lightning_spells.optimizers import Lookahead


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
        callback = RandomAugmentationChoiceCallback(self.callbacks, p=[0.5, 0.5])
        callback.on_train_batch_start(self.trainer, self.module, self.batch, self.batch_idx)
        c1_called = self.callback1.on_train_batch_start.called
        c2_called = self.callback2.on_train_batch_start.called
        # Either one of them is called, but not both
        self.assertTrue(c1_called or c2_called)
        self.assertFalse(c1_called and c2_called)

    def test_no_op_warmup(self):
        # Test that no callback is called during warmup
        callback = RandomAugmentationChoiceCallback(self.callbacks, p=[0.5, 0.5], no_op_warmup=2)
        # First step
        callback.on_train_batch_start(self.trainer, self.module, self.batch, self.batch_idx)
        self.assertFalse(self.callback1.on_train_batch_start.called)
        self.assertFalse(self.callback2.on_train_batch_start.called)
        # Second step
        callback.on_train_batch_start(self.trainer, self.module, self.batch, self.batch_idx)
        self.assertFalse(self.callback1.on_train_batch_start.called)
        self.assertFalse(self.callback2.on_train_batch_start.called)
        # Third step
        callback.on_train_batch_start(self.trainer, self.module, self.batch, self.batch_idx)
        c1_called = self.callback1.on_train_batch_start.called
        c2_called = self.callback2.on_train_batch_start.called
        self.assertTrue(c1_called or c2_called)

    def test_no_op_prob(self):
        # Test that no_op_prob works as expected
        with patch("numpy.random.random", return_value=0.05):
            callback = RandomAugmentationChoiceCallback(self.callbacks, p=[0.5, 0.5], no_op_prob=0.1)
            callback.on_train_batch_start(self.trainer, self.module, self.batch, self.batch_idx)
            self.assertFalse(self.callback1.on_train_batch_start.called)
            self.assertFalse(self.callback2.on_train_batch_start.called)
        with patch("numpy.random.random", return_value=0.15):
            callback = RandomAugmentationChoiceCallback(self.callbacks, p=[0.5, 0.5], no_op_prob=0.1)
            callback.on_train_batch_start(self.trainer, self.module, self.batch, self.batch_idx)
            c1_called = self.callback1.on_train_batch_start.called
            c2_called = self.callback2.on_train_batch_start.called
            self.assertTrue(c1_called or c2_called)


@patch("telegram.Bot")
class TelegramCallbackUnitTest(unittest.TestCase):
    def setUp(self):
        self.token = "test_token"
        self.chat_id = 12345
        self.name = "test_model"
        self.trainer = MagicMock(spec=Trainer)
        self.module = MagicMock()

    def test_on_train_start(self, mock_bot):
        mock_bot.return_value.send_message = AsyncMock()
        callback = TelegramCallback(self.token, self.chat_id, self.name)
        callback.on_train_start(self.trainer, self.module)
        mock_bot.return_value.send_message.assert_called()
        call_args = mock_bot.return_value.send_message.call_args.kwargs
        self.assertIn("has started training", call_args["text"])

    def test_on_train_end(self, mock_bot):
        mock_bot.return_value.send_message = AsyncMock()
        callback = TelegramCallback(self.token, self.chat_id, self.name)
        callback.on_train_end(self.trainer, self.module)
        mock_bot.return_value.send_message.assert_called()
        call_args = mock_bot.return_value.send_message.call_args.kwargs
        self.assertIn("has finished training", call_args["text"])

    def test_on_validation_end(self, mock_bot):
        mock_bot.return_value.send_message = AsyncMock()
        callback = TelegramCallback(self.token, self.chat_id, self.name, report_evals=True)
        self.trainer.logged_metrics = {"val_loss": 0.5}
        self.trainer.global_step = 100
        self.trainer.current_epoch = 1
        callback.on_validation_end(self.trainer, self.module)
        mock_bot.return_value.send_message.assert_called()
        call_args = mock_bot.return_value.send_message.call_args.kwargs
        self.assertIn("Metrics from", call_args["text"])
        self.assertIn("val_loss", call_args["text"])

    def test_report_evals_false(self, mock_bot):
        mock_bot.return_value.send_message = AsyncMock()
        callback = TelegramCallback(self.token, self.chat_id, self.name, report_evals=False)
        self.trainer.logged_metrics = {"val_loss": 0.5}
        callback.on_validation_end(self.trainer, self.module)
        mock_bot.return_value.send_message.assert_not_called()


class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 1)
        self.counter = 0

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.clamp(self(x), 0, 0.1)
        # Use self.counter to ensure the first epoch always get the best val_loss
        loss = self.counter / 10 - torch.nn.functional.mse_loss(y_hat, y)
        self.counter += 1
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        optimizer = Lookahead(optimizer, k=2)
        return optimizer


@patch("telegram.Bot")
class TelegramCallbackIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.train_loader = DataLoader(TensorDataset(torch.randn(10, 32), torch.randn(10, 1)), batch_size=2)
        self.val_loader = DataLoader(TensorDataset(torch.randn(10, 32), torch.randn(10, 1)), batch_size=2)

    def test_telegram_callback(self, mock_bot):
        mock_bot.return_value.send_message = AsyncMock()
        model = SimpleModel()
        callback = TelegramCallback("dummy_token", 1234, "test_model", report_evals=True)
        trainer = Trainer(
            max_epochs=2,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, self.train_loader, self.val_loader)
        assert mock_bot.return_value.send_message.call_count == 4
        for call_args in mock_bot.return_value.send_message.call_args_list:
            print(call_args.kwargs)

        # Training starts
        assert (
            mock_bot.return_value.send_message.call_args_list[0]
            .kwargs["text"]
            .startswith("test_model has started training")
        )
        # First epoch finished
        assert (
            mock_bot.return_value.send_message.call_args_list[1]
            .kwargs["text"]
            .startswith("Metrics from test_model at step 5 (epoch 0)")
        )
        # Second epoch finished
        assert (
            mock_bot.return_value.send_message.call_args_list[2]
            .kwargs["text"]
            .startswith("Metrics from test_model at step 10 (epoch 1)")
        )
        # Training finished
        assert (
            mock_bot.return_value.send_message.call_args_list[3]
            .kwargs["text"]
            .startswith("test_model has finished training 🎉")
        )


NTFY_ENV = {"NTFY_TOPIC": "test_topic", "NTFY_SERVER_URL": "https://ntfy.example.com/", "NTFY_TOKEN": "tk_test"}


def _ntfy_requests(mock_urlopen):
    return [call_args.args[0] for call_args in mock_urlopen.call_args_list]


@patch.dict(os.environ, NTFY_ENV)
@patch("pytorch_lightning_spells.callbacks.urllib.request.urlopen")
class NtfyCallbackUnitTest(unittest.TestCase):
    def setUp(self):
        self.name = "test_model"
        self.trainer = MagicMock(spec=Trainer)
        self.trainer.sanity_checking = False
        self.module = MagicMock()

    def test_config_from_env(self, mock_urlopen):
        callback = NtfyCallback(self.name)
        callback.on_train_start(self.trainer, self.module)
        (request,) = _ntfy_requests(mock_urlopen)
        self.assertEqual(request.full_url, "https://ntfy.example.com/test_topic")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer tk_test")
        self.assertEqual(request.get_header("Priority"), "3")
        self.assertEqual(request.get_header("Title"), "Training started")
        self.assertTrue(request.get_header("User-agent").startswith("pytorch-lightning-spells/"))
        self.assertIn("has started training", request.data.decode("utf-8"))

    def test_missing_topic(self, mock_urlopen):
        with patch.dict(os.environ, {"NTFY_TOPIC": ""}):
            with self.assertRaises(ValueError):
                NtfyCallback(self.name)

    def test_invalid_priority(self, mock_urlopen):
        with patch.dict(os.environ, {"NTFY_PRIORITY": "6"}):
            with self.assertRaises(ValueError):
                NtfyCallback(self.name)

    def test_no_token(self, mock_urlopen):
        with patch.dict(os.environ, {"NTFY_TOKEN": ""}):
            NtfyCallback(self.name).on_train_end(self.trainer, self.module)
        (request,) = _ntfy_requests(mock_urlopen)
        self.assertIsNone(request.get_header("Authorization"))
        self.assertIn("has finished training", request.data.decode("utf-8"))

    def test_on_validation_end(self, mock_urlopen):
        callback = NtfyCallback(self.name, report_evals=True)
        self.trainer.logged_metrics = {"val_loss": 0.5, "epoch": 1}
        self.trainer.global_step = 100
        self.trainer.current_epoch = 1
        callback.on_validation_end(self.trainer, self.module)
        (request,) = _ntfy_requests(mock_urlopen)
        text = request.data.decode("utf-8")
        self.assertIn("at step 100 (epoch 1)", text)
        self.assertIn("val\\_loss: 0.500000", text)
        self.assertNotIn("epoch:", text)

    def test_report_evals_false(self, mock_urlopen):
        callback = NtfyCallback(self.name, report_evals=False)
        self.trainer.logged_metrics = {"val_loss": 0.5}
        callback.on_validation_end(self.trainer, self.module)
        mock_urlopen.assert_not_called()

    def test_skip_sanity_check(self, mock_urlopen):
        callback = NtfyCallback(self.name, report_evals=True)
        self.trainer.sanity_checking = True
        callback.on_validation_end(self.trainer, self.module)
        mock_urlopen.assert_not_called()

    def test_on_exception(self, mock_urlopen):
        NtfyCallback(self.name).on_exception(self.trainer, self.module, RuntimeError("boom"))
        (request,) = _ntfy_requests(mock_urlopen)
        self.assertEqual(request.get_header("Title"), "Training failed")
        self.assertIn("boom", request.data.decode("utf-8"))

    def test_truncate(self, mock_urlopen):
        NtfyCallback(self.name).send_message("é" * 5000, title="t", tags="x")
        (request,) = _ntfy_requests(mock_urlopen)
        self.assertLessEqual(len(request.data), NtfyCallback.MAX_BODY_BYTES)
        self.assertTrue(request.data.decode("utf-8").endswith(NtfyCallback.TRUNCATION_MARKER))

    def test_errors_do_not_raise(self, mock_urlopen):
        callback = NtfyCallback(self.name)
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://ntfy.example.com/test_topic", 403, "Forbidden", {}, io.BytesIO(b"denied tk_test")
        )
        with self.assertLogs("pytorch_lightning_spells.callbacks", level="WARNING") as logs:
            callback.on_train_start(self.trainer, self.module)
        self.assertIn("HTTP 403", logs.output[0])
        # The token is redacted from the logs
        self.assertNotIn("tk_test", logs.output[0])
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with self.assertLogs("pytorch_lightning_spells.callbacks", level="WARNING"):
            callback.on_train_start(self.trainer, self.module)


@patch.dict(os.environ, NTFY_ENV)
@patch("pytorch_lightning_spells.callbacks.urllib.request.urlopen")
class NtfyCallbackIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.train_loader = DataLoader(TensorDataset(torch.randn(10, 32), torch.randn(10, 1)), batch_size=2)
        self.val_loader = DataLoader(TensorDataset(torch.randn(10, 32), torch.randn(10, 1)), batch_size=2)

    def test_ntfy_callback(self, mock_urlopen):
        model = SimpleModel()
        callback = NtfyCallback("test_model", report_evals=True)
        trainer = Trainer(
            max_epochs=2,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, self.train_loader, self.val_loader)
        titles = [request.get_header("Title") for request in _ntfy_requests(mock_urlopen)]
        # The sanity check round is skipped
        self.assertEqual(
            titles, ["Training started", "Validation finished", "Validation finished", "Training finished"]
        )
        texts = [request.data.decode("utf-8") for request in _ntfy_requests(mock_urlopen)]
        self.assertIn("at step 5 (epoch 0)", texts[1])
        self.assertIn("at step 10 (epoch 1)", texts[2])


class LookaheadIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.train_loader = DataLoader(TensorDataset(torch.randn(10, 32), torch.randn(10, 1)), batch_size=2)
        self.val_loader = DataLoader(TensorDataset(torch.randn(10, 32), torch.randn(10, 1)), batch_size=2)

    def test_lookahead_callback(self):
        model = SimpleModel()
        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=3,  # Run more steps for weights to diverge
            limit_val_batches=1,
            callbacks=[LookaheadCallback()],
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, self.train_loader, self.val_loader)

        # Check that the optimizer is a Lookahead optimizer
        self.assertIsInstance(trainer.optimizers[0], Lookahead)

        # Check that the weights are the fast weights after training
        fast_weights = model.layer.weight.clone()

        # Manually trigger validation hooks to check weight swapping
        trainer.callbacks[0].on_validation_start(trainer, model)  # type: ignore
        slow_weights = model.layer.weight.clone()
        self.assertFalse(torch.allclose(fast_weights, slow_weights))

        trainer.callbacks[0].on_validation_end(trainer, model)  # type: ignore
        restored_fast_weights = model.layer.weight.clone()
        self.assertTrue(torch.allclose(fast_weights, restored_fast_weights))

    def test_lookahead_model_checkpoint(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            checkpoint_callback = LookaheadModelCheckpoint(
                dirpath=tmpdir, filename="test", save_top_k=1, monitor="val_loss"
            )
            trainer = Trainer(
                max_epochs=3,
                limit_train_batches=3,  # Run more steps for weights to diverge
                limit_val_batches=1,
                callbacks=[checkpoint_callback],
                logger=False,
            )
            trainer.fit(model, self.train_loader, self.val_loader)

            # Load the checkpoint and check the weights
            self.assertTrue(os.path.exists(checkpoint_callback.best_model_path))
            loaded_model = SimpleModel.load_from_checkpoint(checkpoint_callback.best_model_path)

            # The saved weights should be the slow weights, which are different from the final fast weights
            final_fast_weights = model.layer.weight.clone().cpu()
            saved_slow_weights = loaded_model.layer.weight.clone().cpu()
            print(final_fast_weights, saved_slow_weights)
            self.assertFalse(torch.allclose(final_fast_weights, saved_slow_weights))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SnapMix test requires a GPU")
class CudaAugmentationCallbackTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_classes = 10
        self.input_size = (3, 32, 32)
        self.inputs = torch.randn(self.batch_size, *self.input_size).cuda()
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,)).cuda()
        self.batch = [self.inputs, self.targets]
        self.trainer = MagicMock(spec=Trainer)
        self.module = MagicMock()

    def test_snapmix_callback(self):
        model = MockModel().cuda()
        callback = SnapMixCallback(model, image_size=32, alpha=5.0)
        original_batch = self.batch[0].clone()
        callback.on_train_batch_start(self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, (self.batch_size, 4))
        self.assertFalse(torch.allclose(new_batch, original_batch))


class DeviceAgnosticAugmentationCallbackTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.num_classes = 10
        self.input_size = (3, 32, 32)
        self.inputs = torch.randn(self.batch_size, *self.input_size).to(self.device)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        self.batch = [self.inputs, self.targets]
        self.trainer = MagicMock(spec=Trainer)
        self.module = MagicMock()

    def test_mixup_callback_softmax(self):
        callback = MixUpCallback(alpha=0.4, softmax_target=True)
        original_batch = self.batch[0].clone()
        callback.on_train_batch_start(self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, (self.batch_size, 3))
        self.assertFalse(torch.allclose(new_batch, original_batch))

    def test_mixup_callback_no_softmax(self):
        callback = MixUpCallback(alpha=0.4, softmax_target=False)
        original_batch = self.batch[0].clone()
        original_targets = self.batch[1].clone()
        callback.on_train_batch_start(self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, original_targets.shape)
        self.assertFalse(torch.allclose(new_batch, original_batch))

    def test_cutmix_callback_softmax(self):
        callback = CutMixCallback(alpha=1.0, softmax_target=True)
        original_batch = self.batch[0].clone()
        callback.on_train_batch_start(self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, (self.batch_size, 3))
        self.assertFalse(torch.allclose(new_batch, original_batch))

    def test_cutmix_callback_no_softmax(self):
        callback = CutMixCallback(alpha=1.0, softmax_target=False)
        original_batch = self.batch[0].clone()
        original_targets = self.batch[1].clone()
        callback.on_train_batch_start(self.trainer, self.module, self.batch, 0)
        new_batch, new_targets = self.batch
        self.assertEqual(new_batch.shape, original_batch.shape)
        self.assertEqual(new_targets.shape, original_targets.shape)
        self.assertFalse(torch.allclose(new_batch, original_batch))
