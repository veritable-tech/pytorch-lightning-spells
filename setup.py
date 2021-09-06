#!/usr/bin/env python
import os

from setuptools import find_packages, setup

try:
    from pytorch_lightning_spells import version as pls
except (ImportError, ModuleNotFoundError):
    # alternative https://stackoverflow.com/a/67692/4521646
    import sys
    sys.path.append("pytorch_lightning_spells")
    import version as pls

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pytorch-lightning-spells",
    version=pls.__version__,
    description=pls.__docs__,
    author=pls.__author__,
    author_email=pls.__author_email__,
    url='https://github.com/veritable-tech/pytorch-lightning-spells',
    download_url='https://github.com/veritable-tech/pytorch-lightning-spells',
    license=pls.__license__,
    packages=find_packages(exclude=['tests', 'tests/*', 'benchmarks']),
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=["pytorch_lightning>=1.1.3, <=1.4.0", "scipy", "scikit-learn"],
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
