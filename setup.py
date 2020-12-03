#! /usr/bin/env python

import os
import sys

import setuptools
from setuptools import find_packages

import numpy
from numpy.distutils.core import setup


DISTNAME = 'lightrf'
DESCRIPTION = "Light Random Forest"
URL = 'https://github.com/AaronX121/Light-Random-Forest'
MAINTAINER = 'Yi-Xuan Xu'
MAINTAINER_EMAIL = 'xuyx@lamda.nju.edu.cn'
LICENSE = 'new BSD'
VERSION = '1.0.0'


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('lightrf')

    return config

if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)
    
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          packages=find_packages(),
          include_package_data=True,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          zip_safe=False,
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
            ],
          python_requires=">=3.6",
          install_requires=[
              "numpy",
              "scipy",
              "joblib",
              "Cython>=0.28.5",
              "scikit-learn>=0.22",
          ],
          setup_requires=["cython"])
