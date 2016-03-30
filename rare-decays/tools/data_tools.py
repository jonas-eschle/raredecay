# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:53:18 2016

@author: mayou

Contains several tools to convert, load, save and plot data
"""

import cPickle as pickle
from root_numpy import root2array,rec2array
import pandas as pd
import numpy as np
import config as cfg


def to_pandas(data_in, tree=None, indices=None, columns=None, dtype=None):
    """Convert data from numpy or root to pandas dataframe.

    Converts data safely to pandas, whatever the format is
    """
    is_root = False
    is_ndarray = False
    if (type(data_in) is str):
        is_root = data_in[-5:].upper() in ('.ROOT')
    if is_root:
        data_in = root2array(data_in, treename=tree,branches=columns)
    is_ndarray = type(data_in) is np.ndarray
    if is_ndarray:
        data_in = pd.DataFrame(data_in)
    elif type(data_in) is not pd.core.frame.DataFrame:
        raise TypeError("Could not convert data to pandas. Data: " + data_in)

    return data_in

def adv_return(return_value,logger, save_name=None, multithread=False):

    if save_name not in (None, False):
        if type(save_name) is (str):
            save_name = cfg.PICKLE_PATH + save_name + "." +cfg.PICKLE_DATATYPE
            with open(str(save_name),'wb') as f:
                pickle.dump(return_value, f, pickle.HIGHEST_PROTOCOL)
                logger.info("Data pickled to " + save_name)
        else:
            logger.error("Could not pickle data, name for file (" +
                         str(save_name) + ") is not a string!" +
                         "\n Therefore, the following data was only returned" +
                         " but not saved! \n Data:" + str(return_value))
    return return_value








if __name__ == '__main__':
    print "running selftest"
    dataset = '../data/DarkBoson/HIGGSsignal.root'
    print type(dataset)
#    dataset = root2array(dataset)
    t = to_pandas(dataset)
    print type(t)
    print (type(t) is pd.core.frame.DataFrame)
    print "selftest completed!"
