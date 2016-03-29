# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:49:45 2016

@author: mayou

Contains the different run-modes for the machine-learning algorithms.
"""
import ml_analysis

def run(runmode):
    """select the right runmode from the parameter and run it"""
    print "1,2,3..."
    _test()




def _test():
    print "hello world"
    ml_ana = ml_analysis.MachineLearningAnalysis()
    ml_ana.reweight_mc_real('gb')



# temporary:
if __name__ == '__main__':
    run(1)
