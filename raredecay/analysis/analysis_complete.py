# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:41:40 2016

@author: mayou
"""

from __future__ import division, absolute_import

import raredecay.run_config.config as cfg
from raredecay.globals_ import set_output_handler
set_output_handler(internal=False)
from raredecay.globals_ import out

from raredecay.tools import dev_tool
logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)
out.make_me_a_logger()  # creates a logger inside of "out"

import raredecay.analysis.ml_analysis as ml_ana
import raredecay.analysis.physical_analysis as phys_ana


def output_handler(func):
    """Decorator for output handling"""
    out.initialize()

    out.finalize()

def reweightingKFold():
    out.initialize()

    out.finalize()