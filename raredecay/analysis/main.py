# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:27:14 2016

@author: mayou

Main routine to start the analysis. It contains as few statements as possible.
"""


import matplotlib.pyplot as plt
import cProfile as profile
import seaborn as sns

from raredecay.tools.dev_tool import play_sound
from raredecay.analysis.physical_analysis import run, finalize


print "starting main"
sns.set_context("talk")

# possible loop over method
for i in range(1):
    print "run number ", i+1, " started"
    run(i)
    # show()
finalize()  # finish the analysis, write output etc

# to hear/see whether the analysis has finished
try:
    play_sound()
except:
    print "BEEEEEP"
a = raw_input(["Run finished, press Enter to show the plots"])
plt.show()
