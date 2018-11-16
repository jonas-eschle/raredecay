# -*- coding: utf-8 -*-
"""

@author: Jonas Eschle "Mayou36"


DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat

DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!


Contains several tools to convert, load, save and plot data
"""
# Python 2 backwards compatibility overhead START
from __future__ import division, absolute_import, print_function, unicode_literals

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct,  # noqa
                      open, pow, range, round, str, super, zip,
                      )  # noqa
import sys  # noqa
import warnings  # noqa
import raredecay.meta_config  # noqa

try:  # noqa
    from future.builtins.disabled import (apply, cmp, coerce, execfile, file, long, raw_input,  # noqa
                                          reduce, reload, unicode, xrange, StandardError,
                                          )  # noqa
    from future.standard_library import install_aliases  # noqa

    install_aliases()  # noqa
    from past.builtins import basestring  # noqa
except ImportError as err:  # noqa
    if sys.version_info[0] < 3:  # noqa
        if raredecay.meta_config.SUPPRESS_FUTURE_IMPORT_ERROR:  # noqa
            raredecay.meta_config.warning_occured()  # noqa
            warnings.warn("Module future is not imported, error is suppressed. This means "  # noqa
                          "Python 3 code is run under 2.7, which can cause unpredictable"  # noqa
                          "errors. Best install the future package.", RuntimeWarning)  # noqa
        else:  # noqa
            raise err  # noqa
    else:  # noqa
        basestring = str  # noqa

# Python 2 backwards compatibility overhead END

import warnings
import os
import copy

import pandas as pd
import numpy as np

try:
    import pickle as pickle
except ImportError:
    import pickle

try:
    from root_numpy import array2root
    import root_numpy

except ImportError as err:
    warnings.warn("could not import from root_numpy!")

# from root_numpy import root2array, array2root  # HACK

from raredecay.tools import dev_tool

# both produce error (27.07.2016) when importing them if run from main.py.
# No problem when run as main...

# from raredecay.tools import dev_tool
import raredecay.meta_config as meta_cfg


def apply_cuts(signal_data, bkg_data, percent_sig_to_keep=100, bkg_length=None):
    """Search for best cut on value to still keep percent_sig_to_keep of signal


    Parameters
    ----------
    signal_data : 1-D numpy array
        The signal
    bkg_data : 1-D numpy array
        The background data
    percent_sig_to_keep : 0 < float <= 100
        What percentage of the data to keep in order to apply the cuts.
    """
    #    if percent_sig_to_keep < 100:
    #        raise NotImplementedError("percentage of < 100 not yet imlemented")
    percentile = [0, percent_sig_to_keep]  # TODO: modify for percent_sig_to_keep
    bkg_length_before = len(bkg_data)
    bkg_length = len(bkg_data) if bkg_length in (None, 0) else bkg_length

    lower_cut, upper_cut = np.percentile(signal_data, percentile)
    cut_bkg = np.count_nonzero(np.logical_or(bkg_data < lower_cut, bkg_data > upper_cut))
    rejected_bkg = (bkg_length_before - cut_bkg) / bkg_length

    return [lower_cut, upper_cut], rejected_bkg


def make_root_dict(path_to_rootfile, tree_name, branches):
    """Returns a root_numpy compatible "root-dict" of a root-tree.

    Parameters
    ----------
    path_to_rootfile : str
        The exact path to the root-tree including the filename. Example:
        /home/user1/data/myRootTree1.root
    tree_name : str
        The name of the tree
    branches : str or list[str, str, str,... ]
        The branches of the tree to use
    """
    output = dict(filenames=path_to_rootfile,
                  treename=tree_name,
                  branches=branches)
    output = dev_tool.entries_to_str(output)
    return output


def add_to_rootfile(rootfile, new_branch, branch_name=None, overwrite=True):
    """Adds a new branch to a given root file.

    .. warning:: Overwrite not working currently!


    Parameters
    ----------
    rootfile : root-dict
        The ROOT-file where the data should be added
    new_branch : numpy.array 1-D, list, root-dict
        A one-dimensional numpy array that contains the data.
    branch_name : str
        The name of the branche resp. the name in the dtype of the array.
    """
    from root_numpy import root2array, array2tree

    from rootpy.io import root_open

    rootfile = dev_tool.entries_to_str(rootfile)
    new_branch = dev_tool.entries_to_str(new_branch)
    branch_name = dev_tool.entries_to_str(branch_name)

    # get the right parameters
    # TODO: what does that if there? an assertion maybe?
    write_mode = 'update'
    branch_name = 'new_branch1' if branch_name is None else branch_name

    if isinstance(rootfile, dict):
        filename = rootfile.get('filenames')
    treename = rootfile.get('treename')
    new_branch = to_ndarray(new_branch)
    #    new_branch.dtype = [(branch_name, 'f8')]

    # write to ROOT-file
    write_to_root = False

    if os.path.isfile(filename):
        with root_open(filename, mode='a') as root_file:
            tree = getattr(root_file, treename)  # test
            if not tree.has_branch(branch_name):
                write_to_root = True
    # array2tree(new_branch, tree=tree)
    #            f.write("", TObject.kOverwrite)  # overwrite, does not create friends
    else:
        write_mode = 'recreate'
        write_to_root = True
    if write_to_root:
        arr = np.core.records.fromarrays([new_branch], names=branch_name)
        array2root(arr=arr, filename=filename, treename=treename, mode=write_mode)
        return 0
    else:
        return 1


# TODO: remove? outdated
def format_data_weights(data_to_shape, weights):
    """Format the data and the weights perfectly. Same length and more.

    Change the data to pandas.DataFrame and fill the weights with ones where
    nothing or None is specified. Returns both in lists.
    Very useful to loop over several data and weights.

    Parameters
    ----------
    data_to_shape : (root_dict, numpy.array, pandas.DataFrame)
        The data for which we apply the weights. Usual 2-D shape.
    weights : (list, numpy.array, pandas.DataFrame, None)
        The weights to be reshaped

        *Best format* :

        [array(weights),array(weights), None, array(weights),...]

        *None* can be used if no special weights are specified.
        If weights contains less "weight-containing array-like objects" then
        data_to_shape does, the difference will be filled with *1*

    Return
    ------
    out : list(pandas.DataFrame(data), pandas.DataFrame(data),...)
        Return a list containing data
    out : list(numpy.array(weight), numpy.array(weight),...)
        Return a list with the weights, converted and filled.
    """
    # conver the data
    if not isinstance(data_to_shape, list):
        data_to_shape = [data_to_shape]
    data_to_shape = list(map(to_pandas, data_to_shape))
    # convert the weights
    if not isinstance(weights, list):
        weights = [weights]
    if weights[0] is not None:
        if len(weights[0]) == 1:
            weights = [weights]
    # convert to pandas
    assert isinstance(weights, list), "weights could not be converted to list"
    for data_id, data in enumerate(data_to_shape):
        if data_id >= len(weights):
            weights.append(None)
        if weights[data_id] is None:
            weights[data_id] = np.array([1] * len(data))
        weights[data_id] = to_pandas(weights[data_id]).squeeze().values
    return data_to_shape, weights


def obj_to_string(objects, separator=None):
    """Return a string containing all objects as strings, separated by the separator.

    Useful for automatic conversion for different types. The following objects
    will automatically be converted:

    - None will be omitted

    Parameters
    ----------
    objects : any object or list(obj, obj, ...) with a string representation
        The objects will be converted to a string and concatenated, separated
        by the separator.
    separator : str
        The separator between the objects. Default is " - ".
    """
    objects = dev_tool.entries_to_str(objects)
    if isinstance(objects, basestring):  # no need to change things
        return objects
    separator = " - " if separator is None else separator
    assert isinstance(separator, basestring), "Separator not a string"

    objects = to_list(objects)
    objects = [str(obj) for obj in objects if obj not in (None, "")]  # remove Nones
    string_out = ""
    for word in objects:
        string_out += word + separator if word != objects[-1] else word

    return string_out


def is_root(data_to_check):
    """Check whether a given data is a root file. Needs dicts to be True."""
    flag = False
    data_to_check = dev_tool.entries_to_str(data_to_check)
    if isinstance(data_to_check, dict):
        path_name = data_to_check.get('filenames')
        #        assert isinstance(path_name, str), ("'filenames' of the dictionary " +
        #                                            str(data_to_check) + "is not a string")
        if path_name.endswith(meta_cfg.ROOT_DATATYPE):
            flag = True
    return flag


def is_list(data_to_check):
    """ Check whether the given data is a list."""
    flag = False
    if isinstance(data_to_check, list):
        flag = True
    return flag


def is_ndarray(data_to_check):
    """Check whether a given data is an ndarray."""
    flag = False
    if isinstance(data_to_check, np.ndarray):
        flag = True
    return flag


def is_pickle(data_to_check):
    """Check if the file is a pickled file (checks the ending)."""
    flag = False
    data_to_check = dev_tool.entries_to_str(data_to_check)
    if isinstance(data_to_check, basestring):
        if data_to_check.endswith(meta_cfg.PICKLE_DATATYPE):
            flag = True
    return flag


def to_list(data_in):
    """Convert the data into a list. Does not pack lists into a new one.

    If your input is, for example, a string or a list of strings, or a
    tuple filled with strings, you have, in general, a problem:

    - just iterate through the object will fail because it iterates through the
      characters of the string.
    - using list(obj) converts the tuple, leaves the list but splits the strings
      characters into single elements of a new list.
    - using [obj] creates a list containing a string, but also a list containing
      a list or a tuple, which you did not want to.

    Solution: use to_list(obj), which creates a new list in case the object is
    a single object (a string is a single object in this sence) or converts
    to a list if the object is already a container for several objects.

    Parameters
    ----------
    data_in : any obj
        So far, any object can be entered.

    Returns
    -------
    out : list
        Return a list containing the object or the object converted to a list.
    """
    if isinstance(data_in, (basestring, int, float)):
        data_in = [data_in]
    data_in = list(data_in)
    return data_in


def to_ndarray(data_in, float_array=False):
    """Convert data to numpy array (containing only floats).

    Parameters
    ----------
    data_in : any reasonable data
        The data to be converted
    """
    if is_root(data_in):
        data_in = root_numpy.root2array(**data_in)  # why **? it's a root dict
    # change numpy.void to normal floats
    if isinstance(data_in, (pd.Series, pd.DataFrame)):
        test_sample = data_in.iloc[0]
    else:
        test_sample = data_in[0]
    if isinstance(test_sample, np.void):
        data_in = np.array([val[0] for val in data_in])
    if isinstance(data_in, (np.recarray, np.ndarray)):
        data_in = data_in.tolist()
    if is_list(data_in) or isinstance(data_in, pd.Series):
        data_in = np.array(data_in)
    if not isinstance(data_in[0], (int, float, basestring, bool)):
        if float_array:
            iter_data = copy.deepcopy(data_in)
            # HACK
            data_in = np.ndarray(shape=len(data_in), dtype=data_in.dtype)
            # HACK END
            for i, element in enumerate(iter_data):
                if not isinstance(element, (int, float, basestring, bool)):
                    # does that work or should we iterate over copy?
                    try:
                        element_len = len(element)
                    except TypeError:
                        element_len = 1
                    if element_len > 1:
                        data_in[i] = to_ndarray(element)
                        float_array = False
                    elif element_len == 1:
                        data_in[i] = float(element)

            warnings.warn("Could not force float array")

    if float_array:
        data_in = np.asfarray(data_in)
    assert is_ndarray(data_in), "Error, could not convert data to numpy array"
    return data_in


def to_pandas_old(data_in, index=None, columns=None):
    """Convert data from numpy or root to pandas dataframe.

    Convert data safely to pandas, whatever the format is.
    Parameters
    ----------
    data_in : any reasonable data
        The data to be converted
    """
    # TODO: generalize
    root_index_name = '__index__'

    data_in = dev_tool.entries_to_str(data_in)
    if is_root(data_in):
        root_index = None
        if root_index_name in root_numpy.list_branches(filename=data_in['filenames'],
                                                       treename=data_in.get('treename')):
            root_index = root_numpy.root2array(filenames=data_in['filenames'], treename=data_in.get('treename'),
                                               selection=data_in.get('selection'), branches=root_index_name)
        data_in = root_numpy.root2array(**data_in)  # why **? it's a root dict

    if is_list(data_in):
        data_in = np.array(data_in)
    if is_ndarray(data_in):
        if ((isinstance(columns, (list, tuple)) and len(columns) == 1) or
                isinstance(columns, basestring)):

            data_in = to_ndarray(data_in)
        data_in = pd.DataFrame(data_in, columns=columns, index=root_index)
        if index is not None:
            data_in = data_in.loc[index]
    elif isinstance(data_in, pd.DataFrame):
        pass
    else:
        raise TypeError("Could not convert data to pandas. Data: " + data_in)
    return data_in


def to_pandas(data_in, index=None, columns=None):
    """Convert data from numpy or root to pandas dataframe.

    Convert data safely to pandas, whatever the format is.
    Parameters
    ----------
    data_in : any reasonable data
        The data to be converted
    """
    data_in = dev_tool.entries_to_str(data_in)
    # HACK START
    return to_pandas_old(data_in=data_in, index=index, columns=columns)
    # HACK END
    # from root_pandas import read_root
    #
    # root_pandas_numpy_map = dict(filenames='paths', treename='key', branches='columns',
    #                              selection='where')
    #
    # if is_root(data_in):
    #     is_root2array = False
    #     for key, val in copy.deepcopy(list(data_in.items())):
    #         if key in root_pandas_numpy_map:
    #             is_root2array = True
    #             del data_in[key]
    #             data_in[root_pandas_numpy_map[key]] = val
    #     data_in['columns'] = to_list(data_in['columns'])
    #     if is_root2array:
    #         data_in['columns'] = ['noexpand:'+col for col in data_in['columns'] if not col.startswith('noexpand:')]
    #  remove the noexpand:
    #     data_in = read_root(**data_in)  # why **? it's a root dict
    # if is_list(data_in):
    #     data_in = np.array(data_in)
    # if is_ndarray(data_in):
    #     if ((isinstance(columns, (list, tuple)) and len(columns) == 1) or
    #             isinstance(columns, basestring)):
    #
    #         data_in = to_ndarray(data_in)
    #     data_in = pd.DataFrame(data_in, columns=columns)
    #     if index is not None:
    #         data_in = data_in.loc[index]
    # elif isinstance(data_in, pd.DataFrame):
    #     pass
    # else:
    #     raise TypeError("Could not convert data to pandas. Data: " + data_in)
    # return data_in


def adv_return(return_value, save_name=None):
    """Save the value if save_name specified, otherwise just return input.

    Can be wrapped around the return value. Without any arguments, the return
    of your function will be exactly the same. With arguments, the value can
    be saved (**pickled**) before it is returned.

    Parameters
    ----------
    return_value : any python object
        The python object which should be pickled.
    save_name : str, None
        | The (file-)name for the pickled file. File-extension will be added \
        automatically if specified in *raredecay.meta_config*.
        | If *None* is passed, the object won't be pickled.

    Return
    ------
    out : python object
        Return return_value without changes.

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
    save_name = dev_tool.entries_to_str(save_name)
    if save_name not in (None, False):
        if isinstance(save_name, basestring):
            save_name = meta_cfg.PICKLE_PATH + save_name
            if not is_pickle(save_name):
                save_name += "." + meta_cfg.PICKLE_DATATYPE
            with open(str(save_name), 'wb') as f:
                pickle.dump(return_value, f, meta_cfg.PICKLE_PROTOCOL)
                print(str(return_value) + " pickled to " + save_name)
        else:
            pass
            # HACK how to solve logger problem?
            #            logger.error("Could not pickle data, name for file (" +
            #                         str(save_name) + ") is not a string!" +
            #                         "\n Therefore, the following data was only returned" +
            #                         " but not saved! \n Data:" + str(return_value))
    return return_value


def try_unpickle(file_to_unpickle, use_metapath_bkwcomp=False):
    """Try to unpickle a file and return, otherwise just return input."""
    file_to_unpickle = dev_tool.entries_to_str(file_to_unpickle)
    if is_pickle(file_to_unpickle):
        extra_path = meta_cfg.PICKLE_PATH if use_metapath_bkwcomp else ''
        with open(extra_path + file_to_unpickle, 'rb') as f:
            file_to_unpickle = pickle.load(f)
    return file_to_unpickle
