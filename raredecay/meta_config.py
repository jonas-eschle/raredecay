# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:32:17 2016

@author: mayou

| This module provides the meta-configuration.
| It contains:
 - (package-)global variables for all modules
 - Debug-options which change some implementation on a basic level
 - Global configurations like the endings of specific files etc.

Variables:
---------
run_config:
    It provides the right config module depending on what was chosen
    in the run-methods.
    Should not be changed during the run, only once in the begining.
SUPPRESS_WRONG_SKLEARN_VERSION:
    This package was built for sklearn 0.17. With 0.18 there are some
    module-name changes, which can crash the program.
"""
from __future__ import division

import cPickle as pickle


run_config = None  # 'config'

pathes_to_add = []

# Datatype ending variables
PICKLE_DATATYPE = "pickle"  # default: 'pickle'
ROOT_DATATYPE = "root"  # default 'root'

# DEBUG options
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # default: pickle.HIGHEST_PROTOCOL
MULTITHREAD = False  # not yet implemented
SUPPRESS_WRONG_SKLEARN_VERSION = False
# DON'T CHANGE. Except you know what you do




if __name__ == '__main__':
        # test pathes_to_add
    if not all(type(i) == str for i in pathes_to_add):
        raise TypeError(str(filter(lambda i: type(i) != str, pathes_to_add)) +
                        " not of type string")