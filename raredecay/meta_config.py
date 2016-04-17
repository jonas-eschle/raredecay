# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:32:17 2016

@author: mayou

This module provides the meta-configuration.

Variables:
---------
run_config:
    It provides the right config module depending on what was chosen
    in the run-methods.
SUPPRESS_WRONG_SKLEARN_VERSION:
    This package was built for sklearn 0.17. With 0.18 there are some
    module-name changes, which can crash the program.
"""

global run_config
run_config = 'config'

SUPPRESS_WRONG_SKLEARN_VERSION = False
# DON'T CHANGE. Except you know what you do
