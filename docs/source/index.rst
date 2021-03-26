.. PyTorch Lightning Spells documentation master file, created by
   sphinx-quickstart on Thu Mar 25 14:44:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch Lightning Spells' documentation!
###################################################

This package contains some useful plugins to PyTorch Lightning.

The following is a categorized list of available classes and functions:

CV
***

Augmentation
^^^^^^^^^^^^^

.. warning::   The follwing three callbacks require `MixupSoftmaxLoss <pytorch_lightning_spells.losses.html#pytorch_lightning_spells.losses.MixupSoftmaxLoss>`_ to be used. The target 1-D tensor will be converted to a 2-D one after the callback. The MixupSoftmaxLoss will calculate the correct cross-entropy loss from the 2-D tensor.

* `MixUpCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.MixUpCallback>`_
* `CutMixCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.CutMixCallback>`_
* `SnapMixCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.SnapMixCallback>`_

`RandomAugmentationChoiceCallback <pytorch_lightning_spells.callbacks.html#pytorch_lightning_spells.callbacks.RandomAugmentationChoiceCallback>`_ randomly pick one of the given callbacks for each batch. It also supports a no-op warmup period and setting a no-op probability.

NLP
***

The training and inference speed of NLP models can be improved by sorting the input examples by their lengths.
This reduce the average number of padding tokens per batch (i.e., the input matrices are smaller).
Two samplers are provided to achieve this goal:

* `SortSampler <pytorch_lightning_spells.samplers.html#pytorch_lightning_spells.samplers.SortSampler>`_: Suitable for validation and test dataset, where order of input examples doesn't matter.
* `SortishSampler <pytorch_lightning_spells.samplers.html#pytorch_lightning_spells.samplers.SortishSampler>`_: Suitable for training dataset, where we want to add some randomness in the order of input examples between epochs.

.. toctree::
   :maxdepth: 4
   :caption: Contents:  

   index
   pytorch_lightning_spells



Indices and tables
####################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
