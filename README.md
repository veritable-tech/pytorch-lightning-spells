# PyTorch Lightning Spells

[![CircleCI](https://circleci.com/gh/veritable-tech/pytorch-lightning-spells/tree/main.svg?style=svg)](https://circleci.com/gh/veritable-tech/pytorch-lightning-spells/tree/main) [![Documentation Status](https://readthedocs.org/projects/pytorch-lightning-spells/badge/?version=latest)](https://pytorch-lightning-spells.readthedocs.io/en/latest/?badge=latest)

This package contains some useful plugins for [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

**Disclaimer: This package is a third-party extension for PyTorch Lightning. We are not affiliated with the PyTorch Lightning project or the company behind it.**

[Documentation](https://pytorch-lightning-spells.readthedocs.io/)

## Installation

### PIP

```bash
pip install pytorch-lightning-spells
```

or the latest code in the main branch:

```bash
pip install https://github.com/veritable-tech/pytorch-lightning-spells/archive/main.zip
```

### UV

```bash
uv add pytorch-lightning-spells
```

or the latest code in the main branch:

```bash
uv add git+https://github.com/veritable-tech/pytorch-lightning-spells.git@main
```

## Release Notes

### 0.2.1 (2025-10-08)

A small patch focused on clarity, type-safety, and minor robustness fixes. No public API changes; runtime behavior is preserved.

- Metrics
  - Rename internal state target → targets for clarity (internal only).
  - Add guidance/warnings for AUC and FBeta specialized behavior; refactor compute paths with safer typing.
  - Minor cleanup in SpearmanCorrelation.
- Callbacks
  - TelegramCallback: use event loop run_until_complete, re-enable on_exception, skip sanity validation, read from trainer.logged_metrics, and handle TimedOut gracefully.
  - Lookahead callbacks: assert/operate only on Lookahead optimizers to prevent misuse.
  - RandomAugmentationChoiceCallback: fix typing for p (Sequence[float]) and use random.choices.
- Loggers
  - ScreenLogger: sanitize and pretty-print hyperparams/metrics to ensure JSON-serializable output.
- LR Schedulers
  - Migrate base and multistage schedulers to torch.optim.lr_scheduler.LRScheduler; ensure initial LR setup via _initial_step.
- CI
  - Fix uv sync command and minor workflow hygiene.

Compatibility notes

- No breaking changes expected. Only the internal metric state name changed; user code depending on internal buffers should update from target to targets.

### 0.2.0 (2025-09-03)

- This release modernizes the project's tooling, dependency management, and testing infrastructure.
  - The build system has been migrated from setup.py to the standard pyproject.toml, and the CI/CD pipeline has been overhauled to use uv for significantly faster and more reliable dependency management.
  - It also introduces comprehensive tests for the callback modules, improving code reliability. It also includes several bug fixes and enhancements to existing callbacks, most notably making LookaheadModelCheckpoint more flexible and robust.
- `TelegramCallback`: Update the code to work with the latest version of the `python-telegram-bot` package.
- The `GlobalMetric` class has been updated to align with recent changes in torchmetrics, removing the deprecated compute_on_step parameter.
