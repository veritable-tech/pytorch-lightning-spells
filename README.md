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

### 0.2.0 (2025-09-03)

- This release modernizes the project's tooling, dependency management, and testing infrastructure.
  - The build system has been migrated from setup.py to the standard pyproject.toml, and the CI/CD pipeline has been overhauled to use uv for significantly faster and more reliable dependency management.
  - It also introduces comprehensive tests for the callback modules, improving code reliability. It also includes several bug fixes and enhancements to existing callbacks, most notably making LookaheadModelCheckpoint more flexible and robust.
- `TelegramCallback`: Update the code to work with the latest version of the `python-telegram-bot` package.
- The `GlobalMetric` class has been updated to align with recent changes in torchmetrics, removing the deprecated compute_on_step parameter.
