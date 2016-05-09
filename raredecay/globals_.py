# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2016

@author: mayou

This module contains all (package-)global variables and methods.

Variables:
---------
randint: int
    Many methods need random integers for their pseudo-random generator.
    To keep them all the same (or intentionally not), use the randint.
"""
from __future__ import division, absolute_import

import random

from raredecay.tools import output
from raredecay import meta_config

#==============================================================================
# Output handler. Contains methods "initialize" and "finalize"
#==============================================================================

out = output.OutputHandler()

#==============================================================================
# Random integer generator for pseudo random generator (or other things)
#==============================================================================

randint = random.randint(123, 1512412)  # 357422 or 566575

#==============================================================================
# parallel profile
#==============================================================================

n_cpu_used = 0
def free_cpus():
    n_out = max([meta_config.n_cpu_max - n_cpu_used, 1])
    return n_out




if __name__=='__main__':
    print "Selftest start"
    print "Selftest completed"