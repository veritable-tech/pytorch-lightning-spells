.. PyTorch Lightning Spells documentation master file, created by
   sphinx-quickstart on Thu Mar 25 14:44:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch Lightning Spells' documentation!
###################################################

This package contains some useful plugins for PyTorch Lightning. Many of those are based on others' implementations; I just made some adaptations to make it work with PyTorch Lightning. Please let me know (ceshine at veritable.pw) if you feel any original authors are not credited adequately in my code and documentation.

The following is a categorized list of available classes and functions:

CV
***

Augmentation
^^^^^^^^^^^^^

.. warning:: The following three callbacks require `MixupSoftmaxLoss <pytorch_lightning_spells.losses.html#pytorch_lightning_spells.losses.MixupSoftmaxLoss>`_ to be used. The target 1-D tensor will be converted to a 2-D one after the callback. The MixupSoftmaxLoss will calculate the correct cross-entropy loss from the 2-D tensor.

* `MixUpCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.MixUpCallback>`_
* `CutMixCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.CutMixCallback>`_
* `SnapMixCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.SnapMixCallback>`_

`A notebook is available on Kaggle <https://www.kaggle.com/ceshine/mixup-cutmix-and-snapmix-demo>`_ demonstrating the effect of MixUp, CutMix, and SnapMix.

`RandomAugmentationChoiceCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.RandomAugmentationChoiceCallback>`_ randomly picks one of the given callbacks for each batch. It also supports a no-op warmup period and setting a no-op probability.

NLP
***

The training and inference speed of NLP models can be improved by sorting the input examples by their lengths.
This reduces the average number of padding tokens per batch (i.e., the input matrices are smaller).
Two samplers are provided to achieve this goal:

* `SortSampler <pytorch_lightning_spells.samplers.html#pytorch_lightning_spells.samplers.SortSampler>`_: Suitable for validation and test datasets, where the order of input examples doesn't matter.
* `SortishSampler <pytorch_lightning_spells.samplers.html#pytorch_lightning_spells.samplers.SortishSampler>`_: Suitable for training datasets, where we want to add some randomness in the order of input examples between epochs.


Optimization
*************

* `RAdam <pytorch_lightning_spells.optimizers.html#pytorch_lightning_spells.optimizers.RAdam>`_
* `set_trainable <pytorch_lightning_spells.utils.html#pytorch_lightning_spells.utils.set_trainable>`_: a function that freezes or unfreezes a layer group (a `nn.Module` or `nn.ModuleList`).
* `freeze_layers <pytorch_lightning_spells.utils.html#pytorch_lightning_spells.utils.freeze_layers>`_: a function that freezes or unfreezes **a list of layer groups**.

Lookahead
^^^^^^^^^^^

* `Lookahead <pytorch_lightning_spells.optimizers.html#pytorch_lightning_spells.optimizers.Lookahead>`_: A PyTorch optimizer wrapper to implement the lookahead mechanism.
* `LookaheadCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.LookaheadCallback>`_: A callback that switches the model parameters to the *slow* ones before a validation round starts and switches back to the *fast* ones after it ends.
* `LookaheadModelCheckpoint <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.LookaheadModelCheckpoint>`_: A combination of LookaheadCallback and ModelCheckpoint, so the *slow* parameters are kept in the checkpoints instead of the fast ones.

Learning Rate Schedulers
^^^^^^^^^^^^^^^^^^^^^^^^^

* `MultiStageScheduler <pytorch_lightning_spells.lr_schedulers.html#pytorch_lightning_spells.lr_schedulers.MultiStageScheduler>`_: Allows you to combine several schedulers (e.g., linear warmup and cosine decay).
* `LinearLR <pytorch_lightning_spells.lr_schedulers.html#pytorch_lightning_spells.lr_schedulers.LinearLR>`_: Can be used to achieve both linear warmups and linear decays.
* `ExponentialLR <pytorch_lightning_spells.lr_schedulers.html#pytorch_lightning_spells.lr_schedulers.ExponentialLR>`_

Metrics
********

PyTorch Lightning did not implement metrics that require the entire dataset to have predictions (e.g., AUC, the Spearman correlation). They do have implemented some of them now in the new `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/#indices-and-tables>`_ package.

* `GlobalMetric <pytorch_lightning_spells.metrics.html#pytorch_lightning_spells.metrics.GlobalMetric>`_: Extends this class to create new metrics.
* `AUC <pytorch_lightning_spells.metrics.html#pytorch_lightning_spells.metrics.AUC>`_
* `SpearmanCorrelation <pytorch_lightning_spells.metrics.html#pytorch_lightning_spells.metrics.SpearmanCorrelation>`_
* `FBeta <pytorch_lightning_spells.metrics.html#pytorch_lightning_spells.metrics.FBeta>`_

.. warning:: These metrics require the entire set of labels and predictions to be stored in memory. You might encounter out-of-memory errors if your target tensor is relatively large (e.g., in semantic segmentation tasks) or your validation/test dataset is too large. You'll have to use some approximation techniques in those cases.

Utility
********

* `BaseModule <file:///mnt/SSD_Data/active_projects/pytorch-lightning-spells/docs/build/html/pytorch_lightning_spells.html#module-pytorch_lightning_spells>`_: A boilerplate Lightning Module to be extended upon.
* `ScreenLogger <pytorch_lightning_spells.loggers.html#pytorch_lightning_spells.loggers.ScreenLogger>`_: A logger that prints metrics to the screen.
* `TelegramCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.TelegramCallback>`_: Sent a Telegram message to you when the training starts, ends, and a validation round is finished.
* `EMATracker <pytorch_lightning_spells.utils.html#pytorch_lightning_spells.utils.EMATracker>`_: A exponential moving average aggregator.
* `count_parameters <pytorch_lightning_spells.utils.html#pytorch_lightning_spells.utils.count_parameters>`_: A function that returns the total number of parameters in a model.
* `separate_parameters <pytorch_lightning_spells.utils.html#pytorch_lightning_spells.utils.separate_parameters>`_: A function that split the parameters of a module into two groups (BatchNorm/GroupNorm/LayerNorm and others), so you can use weight decay on only one of them.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Contents:

   index

Indices and tables
####################

* :ref:`modindex`
* :ref:`search`
* :ref:`genindex`