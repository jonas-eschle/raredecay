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
    reweighter1 = ml_ana.reweight_mc_real(meta_cfg=cfg.reweight_meta_cfg, **cfg.reweight_cfg)
    ml_ana.reweight_weights(cfg.reweight_cfg.get('reweight_data_mc'), reweighter1)



# temporary:
if __name__ == '__main__':
    run(1)
