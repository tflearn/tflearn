#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='tflearn',
      version='0.2.1',
      description='Deep Learning Library featuring a higher-level API for '
                  'Tensorflow',
      author='TFLearn contributors',
      author_email='aymeric.damien@gmail.com',
      url='https://github.com/tflearn/tflearn',
      download_url='https://github.com/tflearn/tflearn/tarball/0.2.1',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'six',
          'Pillow'
      ],
      classifiers=[
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Developers'
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      keywords=[
          'TFLearn',
          'TensorFlow',
          'Deep Learning',
          'Machine Learning',
          'Neural Networks',
          'AI'
        ]
      )
