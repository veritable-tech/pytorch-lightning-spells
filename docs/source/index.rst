.. PyTorch Lightning Spells documentation master file, created by
   sphinx-quickstart on Thu Mar 25 14:44:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch Lightning Spells' documentation!
====================================================

This package contains some useful plugins to PyTorch Lightning.

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

   pytorch_lightning_spells



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
