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
out.initialize()
out.add_output("test1")
out.add_output("test2_silent")
print ["1111111111111111"] * 9999999
out.finalize()