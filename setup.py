# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:26:12 2016

@author: Jonas Eschle "Mayou36"
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import io
import os
import subprocess

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()


def readme():
    with open('README.rst') as f:
        return f.read()


git_version = '2.3.0'

if __name__ == '__main__':
    setup(name='raredecay',
          version=git_version,
          description='A package with multivariate analysis and reweighting '
                      'algorithms',
          long_description=readme(),
          classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: Apache Software License',
              'Natural Language :: English',
              'Operating System :: MacOS',
              'Operating System :: MacOS :: MacOS X',
              'Operating System :: POSIX :: Linux',
              'Operating System :: Unix',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: Implementation :: CPython',
              'Topic :: Scientific/Engineering :: Physics',
              'Topic :: Scientific/Engineering :: Information Analysis',
          ],
          keywords='particle physics, analysis, machine learning, reweight, high energy physics',
          url='https://github.com/mayou36/raredecay',
          author='Jonas Eschle',
          author_email='Jonas.Eschle@cern.ch',
          license='Apache-2.0 License',
          install_requires=requirements,
          packages=['raredecay',
                    'raredecay.analysis',
                    'raredecay.tools',
                    ],
          include_package_data=True,
          python_requires=">2.7,!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
          zip_safe=False
          )

    # build docs
    try:
        subprocess.Popen("chmod u+x " + os.path.join(here, 'docs/make_docs.sh'), shell=True)
        subprocess.Popen("bash " + os.path.join(here, 'docs/make_docs.sh'), shell=True)
    except Exception as err:
        print("Failed to build docs.")
        raise err
