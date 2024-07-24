##########################################################################
##  Perfomance Analyzer tools
##
##  This code is the python pacage setup tool for performance-analyzer
##  which contains serveral tools for LPU.
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

from setuptools import setup, find_packages

##########################################################################
## Python Package setup
##########################################################################

setup(
  name='performance-analyzer',
  version='1.3.1',
  description='Python package for analyze hardware performance',
  author='junsoo kim',
  author_email='js.kim@hyperaccel.ai',
  packages=find_packages(),
  install_requires=[
    'transformers>=4.35.0',
    'torch>=2.1.0'
  ]
)

##########################################################################
