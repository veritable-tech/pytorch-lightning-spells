#!/usr/bin/env python
import os

from setuptools import find_packages, setup

import pytorch_lightning_spells

setup(
    name="pytorch-lightning-spells",
    version=pytorch_lightning_spells.__version__,
    description=pytorch_lightning_spells.__docs__,
    author=pytorch_lightning_spells.__author__,
    author_email=pytorch_lightning_spells.__author_email__,
    url='https://github.com/veritable-tech/pytorch-lightning-spells',
    download_url='https://github.com/veritable-tech/pytorch-lightning-spells',
    license=pytorch_lightning_spells.__license__,
    packages=find_packages(exclude=['tests', 'tests/*', 'benchmarks']),

    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=["pytorch_lightning>=1.1.3"],
    extras_require={},
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)