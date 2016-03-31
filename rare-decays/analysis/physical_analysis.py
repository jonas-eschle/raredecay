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
    _test2()




def _test1():
    print "starting physical module test"
    ml_ana = ml_analysis.MachineLearningAnalysis()
    reweighter1 = ml_ana.reweight_mc_real(meta_cfg=cfg.reweight_meta_cfg, **cfg.reweight_cfg)
    new_weights = ml_ana.reweight_weights(cfg.reweight_cfg.get('reweight_data_mc'), reweighter1)
    #new_weights = ml_ana.reweight_weights(cfg.reweight_cfg.get('reweight_data_mc'), "reweighter1.pickl.pickle")
    print new_weights

def _test2():
    
# temporary:
if __name__ == '__main__':
    run(1)
