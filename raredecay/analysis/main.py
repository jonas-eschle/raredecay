# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:27:14 2016

@author: mayou

Main routine to start the analysis. It contains as few statements as possible.
"""
from __future__ import division, absolute_import

import matplotlib.pyplot as plt
import cProfile as profile
import seaborn as sns


from raredecay.analysis.physical_analysis import run


def main_run(run_mode, cfg_file=None):
    # set plotting style
    sns.set_context("poster")
    plt.rc('figure', figsize=(20, 20))

    print "starting main, run: " + run_mode

    if run_mode is None:
        run_mode = "reweight_comparison"
        print "Run mode was None, set to default " + run_mode
    # possible loop over method
    n_executions = 1
    for i in range(n_executions):
        if n_executions > 1:
            print "run number", i+1, "of", n_executions, " started"
        run(run_mode, cfg_file=cfg_file)
        # show()

    # to hear/see whether the analysis has finished
    try:
        from raredecay.tools.dev_tool import play_sound
        play_sound()
    except:
        print "BEEEEEP"
    raw_input(["Run finished, press Enter to show the plots"])
    plt.show()

if __name__ == '__main__':
    #main_run("reweight_comparison")
    #main_run("reweight")
    main_run("reweightCV")
    #main_run("simple_plot")
    #main_run("hyper_optimization")


