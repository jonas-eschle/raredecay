# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:32:17 2016

@author: mayou
"""

global run_config
run_config = 'config'


def init_config(new_config):
    global run_config
    run_config = new_config


output_string = ""


SUPPRESS_WRONG_SKLEARN_VERSION = False
# DON'T CHANGE. Except you know what you do
