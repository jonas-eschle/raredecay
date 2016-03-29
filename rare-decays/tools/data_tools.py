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


def to_pandas(data_in, tree=None, indices=None, columns=None, dtype=None):
    """Convert data from numpy or root to pandas dataframe.

    Converts data safely to pandas, whatever the format is
    """
    is_root = False
    is_ndarray = False
    if (type(data_in) is str):
        is_root = data_in[-5:].upper() in ('.ROOT')
    if is_root:
        data_in = root2array(data_in, treename=tree,branches=indices)
    is_ndarray = type(data_in) is np.ndarray
    if is_ndarray:
        data_in = pd.DataFrame(data_in, columns)
    elif type(data_in) is not pd.core.frame.DataFrame:
        raise TypeError("Could not convert data to pandas. Data: " + data_in)

    return data_in




if __name__ == '__main__':

    dataset = '../data/DarkBoson/HIGGSsignal.rodot'
    print type(dataset)
#    dataset = root2array(dataset)
    t = to_pandas(dataset)
    print type(t)
    print (type(t) is pd.core.frame.DataFrame)