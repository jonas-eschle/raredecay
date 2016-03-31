# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:53:18 2016

@author: mayou

Contains several tools to convert, load, save and plot data
"""

import cPickle as pickle
from root_numpy import root2array, rec2array
import pandas as pd
import numpy as np
import config as cfg


def is_root(data_to_check):
    flag = False
    """Check whether a given data is a root file. Needs dicts to be True!
    """
    if type(data_to_check) is dict:
        path_name = data_to_check.get('filenames')
        assert type(path_name) is str, ("'filenames' of the dictionary " +
                                        data_to_check + "is not a string")
        if path_name.endswith(cfg.ROOT_DATATYPE):
            flag = True
    return flag


def is_ndarray(data_to_check):
    flag = False
    """Check whether a given data is an ndarray.
    """
    if type(data_to_check) is np.ndarray:
        flag = True
    return flag


def is_pickle(data_to_check):
    flag = False
    if type(data_to_check) is str:
        if data_to_check.endswith(cfg.PICKLE_DATATYPE):
            flag = True
    return flag


def to_pandas(data_in, indices=None, columns=None, dtype=None):
    """Convert data from numpy or root to pandas dataframe.

    Convert data safely to pandas, whatever the format is
    """

    if is_root(data_in):
        data_in = root2array(**data_in)
    if is_ndarray(data_in):
        data_in = pd.DataFrame(data_in)
    elif type(data_in) is not pd.core.frame.DataFrame:
        pass
    else:
        raise TypeError("Could not convert data to pandas. Data: " + data_in)
    return data_in


def adv_return(return_value, logger, save_name=None, multithread=False):

    if save_name not in (None, False):
        if type(save_name) is str:
            save_name = cfg.PICKLE_PATH + save_name
            if not is_pickle(save_name):
                save_name += "." + cfg.PICKLE_DATATYPE
            with open(str(save_name), 'wb') as f:
                pickle.dump(return_value, f, cfg.PICKLE_PROTOCOL)
                logger.info("Data pickled to " + save_name)
        else:
            logger.error("Could not pickle data, name for file (" +
                         str(save_name) + ") is not a string!" +
                         "\n Therefore, the following data was only returned" +
                         " but not saved! \n Data:" + str(return_value))
    return return_value


def try_unpickle(file_to_unpickle):
    if is_pickle(file_to_unpickle):
        with open(cfg.PICKLE_PATH + file_to_unpickle, 'rb') as f:
            file_to_unpickle = pickle.load(f)
    return file_to_unpickle


if __name__ == '__main__':
    print "running selftest"
    dataset = {'filenames': '../data/DarkBoson/HIGGSsignal.root'}
    print type(dataset)
#    dataset = root2array(dataset)
    t = to_pandas(dataset)
    print type(t)
    print (type(t) is pd.core.frame.DataFrame)
    print "selftest completed!"
