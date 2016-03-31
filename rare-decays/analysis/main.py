# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:27:14 2016

@author: mayou

Main routine to run the analysis
"""

from dev_tool import play_sound
from physical_analysis import run
import cProfile as profile
print "starting main"
run(1)
play_sound()
