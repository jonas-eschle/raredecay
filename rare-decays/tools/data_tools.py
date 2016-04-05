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
#import config as cfg
import importlib
import meta_config
cfg = importlib.import_module(meta_config.run_config)

import dev_tool

module_logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)

def apply_weights(data_to_apply, weights, logger=None,
                  ml_analysis_object=None):
    """Reweight the data with given weights.

    Applies the weights on the data. Formats can nearly be arbitrary chosen.

    Parameters
    ----------
    data_to_apply : (root_dict, numpy.array, pandas.DataFrame)
        The data for which we apply the weights. Usual 2-D shape.
    weights : (list, numpy.array, pandas.DataFrame, None)
        The weights to be applied.

        *Best format* :

        [array(weights),array(weights), None, array(weights),...]

        *None* can be used if no special weights are specified.
        If weights contains less weight-containing array-like objects then
        data_to_apply does, the difference will be filled with *None*

    Return
    ------
    out : list(pandas.DataFrame(data), pandas.DataFrame(data),...)
        Return a list containing data with the new weights applied
    out : list(pandas.DataFrame(weight), pandas.DataFrame(weight),...)
        Return a list with the weights, converted and filled.
    """
    if logger is None:
        logger = module_logger
    # conver the data
    if not isinstance(data_to_apply, list):
        data_to_apply = [data_to_apply]
    logger.debug("data_to_apply: " + str(data_to_apply))
    if (hasattr(ml_analysis_object, '__getitem__') and  # could also be 'None'
            hasattr(ml_analysis_object, 'fast_to_pandas')):
        data_to_apply = map(ml_analysis_object.fast_to_pandas, data_to_apply)
    else:
        data_to_apply = map(to_pandas, data_to_apply)
    # convert the weights
    if not isinstance(weights, list):
        weights = [weights]
    logger.debug("weights: " + str(weights))
    if weights[0] is not None:
        if len(weights[0]) == 1:
            weights = [weights]
    # convert to pandas and multiply all data with the weights
    assert isinstance(weights, list), "weights could not be converted to list"
    for data_id, data in enumerate(data_to_apply):
        if data_id >= len(weights):
            weights.append(None)
        if weights[data_id] is None:
            weights[data_id] = np.array([1] * len(data))
        weights[data_id] = to_pandas(weights[data_id]).squeeze().values
        data_to_apply[data_id].multiply(weights[data_id], axis=0)
    return data_to_apply, weights


def is_root(data_to_check):
    """Check whether a given data is a root file. Needs dicts to be True!
    """
    flag = False
    if type(data_to_check) is dict:
        path_name = data_to_check.get('filenames')
        assert type(path_name) is str, ("'filenames' of the dictionary " +
                                        data_to_check + "is not a string")
        if path_name.endswith(cfg.ROOT_DATATYPE):
            flag = True
    return flag


def is_list(data_to_check):
    """ Check whether the given data is a list
    """
    flag = False
    if isinstance(data_to_check, list):
        flag = True
    return flag


def is_ndarray(data_to_check):
    """Check whether a given data is an ndarray.
    """
    flag = False
    if type(data_to_check) is np.ndarray:
        flag = True
    return flag


def is_pickle(data_to_check):
    flag = False
    if type(data_to_check) is str:
        if data_to_check.endswith(cfg.PICKLE_DATATYPE):
            flag = True
    return flag


def to_pandas(data_in, logger=None, indices=None, columns=None, dtype=None):
    """Convert data from numpy or root to pandas dataframe.

    Convert data safely to pandas, whatever the format is.
    """
    if logger is None:
        logger = module_logger
    if is_root(data_in):
        data_in = root2array(**data_in)  # why **? it's a root dict
    if is_list(data_in):
        data_in = np.array(data_in)
    if is_ndarray(data_in):
        data_in = pd.DataFrame(data_in)
    elif type(data_in) is pd.core.frame.DataFrame:
        pass
    else:
        raise TypeError("Could not convert data to pandas. Data: " + data_in)
    return data_in


def adv_return(return_value, logger=None, save_name=None, multithread=False):
    """Save the value if save_name specified, otherwise just return input

    Can be wrapped around the return value. Without any arguments, the return
    of your function will be exactly the same. With arguments, the value can
    be saved (**pickled**) before it is returned.

    **Usage**:
     Instead of a simple return statement

     >>> return my_variable/my_object

     one can use the **completely equivalent** statement

     >>> return adv_return(my_variable/my_object)

     If the return value should be saved in addition to be returned, use

     >>> return adv_return(my_variable/my_object, save_name='my_object.pickle')

      (*the .pickle ending is not required but added automatically if omitted*)
     which returns the value and saves it.
    """
    if logger is None:
        logger = module_logger
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

    print "selftest completed!"
