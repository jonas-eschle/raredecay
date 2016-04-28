# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2016

@author: mayou

This module contains all (package-)global variables and methods

Variables:
---------
randint: int
    Many methods need random integers for their pseudo-random generator.
    To keep them all the same (or intentionally not), use the randint.
"""
from __future__ import division

import warnings
import random


randint = random.randint(123, 1512412)  # 357422 or 566575

