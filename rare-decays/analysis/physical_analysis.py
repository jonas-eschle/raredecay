# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:49:45 2016

@author: mayou

Contains the different run-modes for the machine-learning algorithms.
"""
import ml_analysis
import config as cfg

def run(runmode):
    """select the right runmode from the parameter and run it"""
    print "1,2,3..."
    _test()




def _test():
    print "starting physical module test"
    ml_ana = ml_analysis.MachineLearningAnalysis()
    print ml_ana.reweight_mc_real(meta_cfg=cfg.reweight_meta_cfg, **cfg.reweight_cfg)



# temporary:
if __name__ == '__main__':
    run(1)
