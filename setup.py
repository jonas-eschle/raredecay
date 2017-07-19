# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:26:12 2016

@author: Jonas Eschle "Mayou36"
"""

from setuptools import setup
import subprocess

import io
import os

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split('\n')


def readme():
    with open('README.md') as f:
        return f.read()
#try:
#    git_version = subprocess.check_output(["git", "-C",
#                        "/home/mayou/Documents/uniphysik/Bachelor_thesis/python_workspace/raredecay",
#                        "describe"])
#    git_version = git_version.partition('-')
#    git_version = str(git_version[0])
#except:
#    git_version = '1.0.2'
git_version = '1.3.0'


setup(name='raredecay',
      version=git_version,
      description='A package for analysis of rare particle decays with machine-learning algorithms',
      long_description=readme(),
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS',
		'Operating System :: MacOS :: MacOS 9',
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: POSIX :: Linux',
		'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 2 :: Only',
		'Programming Language :: Python :: Implementation :: CPython',
		'Topic :: Scientific/Engineering :: Physics',
		'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      keywords='particle physics, analysis, machine learning, reweight, high energy physics',
      url='https://github.com/mayou36/raredecay',
      author='Jonas Eschle',
      author_email='mayou36@jonas.eschle.com',
      license='Apache-2.0 License',
      dependency_links=['https://github.com/yandex/rep/archive/stratifiedkfold.zip'],
      install_requires=requirements,
      extras_require={
                      'root': ['root_numpy',
                               'rootpy']
      		     },
      packages=['raredecay',
                'raredecay.analysis',
                'raredecay.run_config',
                'raredecay.tools',
                ],
      include_package_data=True,
      zip_safe=False
      )
