#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, setuptools


__author__ = 'Iván de Paz Centeno'


def readme():
    with open('README.rst', encoding="UTF-8") as f:
        return f.read()


if sys.version_info < (3, 4, 1):
    sys.exit('Python < 3.4.1 is not supported!')


setup(name='deepevolution',
      version="0.0.1",
      description='A genetic algorithm to evolve tensorflow keras neural networks.',
      long_description=readme(),
      url='http://github.com/ipazc/deepevolution',
      author='Iván de Paz Centeno',
      author_email='ipazc@unileon.es',
      packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
      install_requires=[
          "numpy",
          "pandas>=1.1.0",
          "tqdm"
      ],
      extras_require={
          "tf": ["tensorflow>=2.0.0"],
          "tf_gpu": ["tensorflow-gpu>=2.0.0"],
      },
      classifiers=[
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      include_package_data=True,
      keywords="deepevolution genetic algorithm keras tensorflow evolve network",
      zip_safe=False)
