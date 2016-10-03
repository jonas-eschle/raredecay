# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:10:29 2016

@author: mayou
"""
# debug
from __future__ import division, absolute_import

import copy
import warnings
#import cProfile as profile

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import deque
import itertools

from root_numpy import root2rec
from rep.data.storage import LabeledDataStorage

from raredecay.tools import data_tools, dev_tool
try:
    from raredecay.globals_ import out
    out_imported = True
except ImportError:
    warnings.warn(ImportWarning, "could not import out. Some functions regarding output (save figure etc.) won't be available")
    out_imported = False
from raredecay import meta_config

## TODO: import config not needed??
## import configuration
import importlib
##from raredecay import meta_config
cfg = importlib.import_module(meta_config.run_config)
modul_logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


class HEPDataStorage(object):
    """ A wrapper around pandas.DataFrame and an extension to the
    LabeledDataStorage.



    """
    # define constants for independence
    __ROOT_DATATYPE = meta_config.ROOT_DATATYPE

    __figure_number = 0
    __figure_dic = {}

    def __init__(self, data, index=None, target=None, sample_weights=None,
                 data_name=None, data_name_addition=None, data_labels=None,
                 hist_settings=None, supertitle_fontsize=18):
        """Initialize instance and load data

        Parameters
        ----------
        data : (root-tree dict, pandas DataFrame)

            - **root-tree dict** (*root-dict*):
            |   Dictionary which specifies all the information to convert a root-
            |   tree to an array. Directly given to :py:func:`~root_numpy.root2rec`

            - **pandas DataFrame**:
            |   A pandas DataFrame. The index (if not explicitly defined)
                and column names will be taken.

        index : 1-D array-like
            The indices of the data that will be used.
        target : list or 1-D array or int {0, 1}
            Labels the data for the machine learning. Usually the y.
        sample_weights : 1-D array or {1, None}
            Contains the weights of the samples.
        .. note:: If None or 1 specified, 1 will be assumed for all.
        data_name : str
            | Name of the data, human-readable. Displayed in the title of \
            plots.
            | *Example: 'Bu2K1piee mc', 'beta-decay real data' etc.*
        data_name_addition : str
            | Additional remarks to the data, human readable. Displayed in \
            the title of plots.
            | *Example: 'reweighted', 'shuffled', '5 GeV cut applied' etc.*
        data_labels : dict with strings {column name: human readable name}
            | Human-readable names for the columns, displayed in the plot.
            | Dictionary has to contain the exact column (=branch) name of \
            the data
            | All not specified labels will be auto-labeled by the branch \
              name itself.

            | *Good practice*: keep a dictionary containing all possible lables
              and hand it over every time.
        add_label : boolean
            If true, the human-readable labels will be added to the branch name
            shows in the plot instead of replaced.
        hist_settings : dict
            Dictionary with the settings for the histogram plot function
            :func:`~matplotlip.pyplot.hist`
        supertitle_fontsize : int
            The size of the title of several subplots.
        """
        # initialize logger
        self.logger = modul_logger

        # initialize data
        self._fold_index = None  # list with indeces of folds
        self._fold_status = None  # tuple (my_fold_number, total_n_folds)
        self.set_data(data=data, index=index)

        # data name
        self._name = ["", "", ""]
        data_name = "unnamed data" if data_name is None else data_name
        self.data_name = data_name
        self.data_name_addition = data_name_addition
        self.fold_name = None

        # initialize targets
        self._set_target(target=target)

        # data-labels human readable, initialize with the column name
        self._label_dic = {}
        self._label_dic = {col: col for col in self.columns if self._label_dic.get(col) is None}
        self.set_labels(data_labels=data_labels)

        # initialize weights
        self._weights = None
        self.set_weights(sample_weights)

        # plot settings
        if dev_tool.is_in_primitive(hist_settings, None):
            hist_settings = meta_config.DEFAULT_HIST_SETTINGS
        self.hist_settings = hist_settings
        self.supertitle_fontsize = supertitle_fontsize

    def __len__(self):
        return self._length

# TODO: remove obsolet
    def get_name(self):
        """Return the human-readable name of the data as a string"""
        warnings.warn("Depreceated, will be removed. Use obj.name instead.", FutureWarning)
        return self._get_name()

    @property
    def name(self):
        """Return the **full** human-readable name of the data as a string"""
        return self._get_name()

    def _get_name(self):
        out_str = data_tools.obj_to_string(self._name, separator=" ")
        return out_str

    def _set_name(self, data_name=None, data_name_addition=None, fold_name=None):
        """Set the data name"""
        # set the new name in self._name
        for i, name in enumerate([data_name, data_name_addition, fold_name]):
            if name is not None:
                self._name[i] = name

#TODO: change the naming into a dict
    @property
    def data_name(self):
        return self._name[0]

    @property
    def data_name_addition(self):
        return self._name[1]

    @property
    def fold_name(self):
        return self._name[2]

    @data_name.setter
    def data_name(self, data_name):
        self._set_name(data_name=data_name)

    @data_name_addition.setter
    def data_name_addition(self, data_name_addition):
        self._set_name(data_name_addition=data_name_addition)

    @fold_name.setter
    def fold_name(self, fold_name):
        self._set_name(fold_name=fold_name)

    def get_index(self):
        """Return the index used inside the DataStorage. Advanced feature."""
        warnings.warn("Will be removed in the future. Use obj.index instead", FutureWarning)
        return self._make_index()

    @property
    def index(self):
        """Return the *real* index as a list"""
        return self._make_index()

    @index.setter
    def index(self, index):
        self._set_index(index)

    def _make_index(self, index=None):
        """Return the index, else the self._index. If none exist, **create**
        the normal one

        It has the following priorities:

        1. if the given index is not None, it will be taken
        2. next look for the self._index. If there is one, it will be returned
        3. otherwise, a list of indeces as usuall (0, len-1) will be returned
        """
        if index is None:
            temp = list(range(len(self))) if self._index is None else self._index
        else:
            temp = index
        return temp

    def _set_index(self, index):
        """If index is not None -> assign. Else try to get from data"""
        if index is None:
            self._index = None
            if self._data_type == 'root':
                pass  # no index contained in root-dicts
            elif self._data_type == 'array':
                pass  # no index information contained in an array
            elif self._data_type == 'df':
                index_list = self._data.index.tolist()
                # TODO: remove HACK with length, replace with len(self)
                if not index_list == range(len(self)):  # if special indexing
                    self._index = index_list
        else:
            self._index = index

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        # TODO: maybe check?
        self._set_columns(columns=columns)

    def _set_columns(self, columns):

        if columns is None:
            if self._data_type == 'root':
                self._columns = data_tools.to_list(self._data['branches'])
            elif self._data_type == 'df':
                self._columns = data_tools.to_list(self._data.columns.values)
            elif self._data_type == 'array':
                self._columns = ['feature_' + str(i) for i in range(len(self._data))]
        else:

            self._columns = columns


    def _set_length(self, index):
        # determine whether to set length individually from the data or not
        if index is None:
            if self._data_type == 'root':
                temp_root_dict = copy.deepcopy(self._data)
                temp_branch = temp_root_dict.pop('branches')  # remove to only use one branch
                temp_branch = data_tools.to_list(temp_branch)
                self._length = len(root2rec(branches=temp_branch[0], **temp_root_dict))
            elif self._data_type == 'df':
                self._length = len(self._data)
            elif self._data_type == 'array':
                self._length = self._data.shape[1]

        else:
            self._length = len(index)

    @staticmethod
    def _get_data_type(data):
        """Return the type of the data

        - 'df' : pandas DataFrame
        - 'root': root-file
        - 'array': numpy array
        """
        data_type = None
        if isinstance(data, dict):
            if data.has_key('filenames') and data['filenames'].endswith(HEPDataStorage.__ROOT_DATATYPE):
                data_type = 'root'
        elif isinstance(data, pd.DataFrame):
            data_type = 'df'
        elif isinstance(data, (np.ndarray, np.array)):
            data_type = 'array'

        return data_type

    @property
    def data(self):
        return self._data

    def set_data(self, data, index=None, columns=None):

        self._set_data(data=data, index=index, columns=columns)

    def _set_data(self, data, index=None, columns=None):
        """Set the data, length- and columns-attribute

        Convert the data to the right (root-dict, df etc.) format (and save).
        Also set the length and columns.

        currently implemented:
            - ROOT-data file (*root-dict*)
            - Pandas DataFrame
        """
        # get the data_type
        self._data = data
        self._data_type = self._get_data_type(data)

        self._set_length(index=index)
        self.index = index
        self.columns = columns


        # convert the data (and save it)

        # root data
        if self._data_type == 'root':
            pass
        # pandas DataFrame
        elif self._data_type == 'df':
            self._data = self._make_df(data=self._data, index=self._index)
        # numpy array
        elif self._data_type == 'array':
            self._data = self._make_df(data=data, index=self._index)
        else:
            raise NotImplementedError("Other dataformats are not yet implemented")

# TODO: implement pickleable data?

    @property
    def labels(self):
        return self._label_dic.get(self.columns)



    def get_weights(self, index=None, normalize=True, weights_as_events=False,
                    min_weight=None):
        """Return the weights of the specified indeces or, if None, return all.

        Parameters
        ----------
        normalize : boolean or float > 0
            If True, the weights will be normalized to 1 (the mean is 1).
            No effect, if weights_as_events is True.
            If a float is provided, the mean of the weights will be equal
            to *normalize*. So *True* and *1* will yield the same results.
        index : 1-D list
            List of index to get the weights from.
        weights_as_events : int >= 1 or False, None
            Return the weights converted to number of events. Basically,
            return the weights compatible to the *pandasDF* function with
            the argument *weights_as_events* passed.

            Simple: create a numpy array with ones of the right length.

        min_weight : float or int
            For "weights_as_events"; normaly, the minimum of the weights is
            used to scale, but providing a *min_weight* will use the minimum
            of the two minima for scaling.



        Return
        ------
        out: 1-D numpy array
            Return the weights as pandas Series
        """
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        weights_out = self._get_weights(index=index, normalize=normalize,
                                        weights_as_events=weights_as_events,
                                        min_weight=min_weight)

        if dev_tool.is_in_primitive(weights_out, (None, 1)):
            weights_out = dev_tool.fill_list_var([], length, 1)
            weights_out = np.array(weights_out)
            weights_out = pd.Series(data=weights_out, index=index)
#        else:
#            weights_out = weights_out.as_matrix()
        # TODO: maybe remove:
        #weights_out = data_tools.to_ndarray(weights_out)

        return weights_out

    def _get_weights(self, index=None,  normalize=True, weights_as_events=False, min_weight=None):
        # initialize values
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        if weights_as_events >= 1:
            pass
        elif dev_tool.is_in_primitive(self._weights, (None, 1)):
            weights_out = self._weights
            if normalize != 1 or normalize is not True:
                weights_out = pd.Series(np.ones(length), index=index)
            else:
                normalize = False
        elif index is None:
            weights_out = self._weights
        else:
            weights_out = self._weights.loc[index]

        if weights_as_events >= 1:
            length = sum(self._scale_weights(index=index, weights_as_events=weights_as_events,
                                             cast_int=True, min_weight=min_weight))
            weights_out = pd.Series(np.ones(length))


        elif normalize or normalize > 0:  # explicit is better then implicit
            eps = 0.00001
            counter = 0
            normalize = 1 if normalize is True else normalize
            while not (normalize - eps < weights_out.mean() and weights_out.mean() < normalize + eps):
                weights_out *= normalize/weights_out.mean()
                counter += 1
                if counter > 10:  # to prevent endless loops
                    self.logger.error("Could not normalize weights. Mean(weights): " + str(weights_out.mean()) + " of " + str(normalize))
                    meta_config.error_occured()
                    break
        return weights_out

    def set_weights(self, sample_weights, index=None):
        """Set the weights of the sample.

        Parameters
        ----------
        sample_weights : 1-D array-like or list or int {1} (or str/dict for root-trees)
            The new weights for the dataset. If the data is a root-tree file,
            a string (naming the branche) or a whole root-dict can be given,
            pointing to the weights stored.
        index : 1-D array or list or None
            The indeces for the weights to be set
        """
        index = self._index if index is None else index
        length = len(self) if index is None else len(index)

        if isinstance(sample_weights, (str, dict)) and self._data_type == 'root':
            assert isinstance(self._data, dict), "data should be root-dict but is no more..."
            tmp_root = copy.deepcopy(self._data)
            if isinstance(sample_weights, str):
                sample_weights = {'branches': sample_weights}
            tmp_root.update(sample_weights)
            branche = tmp_root['branches']
            assert (isinstance(branche, list) and len(branche) == 1) or isinstance(branche, str), "Can only be one branche"
            sample_weights = data_tools.to_ndarray(tmp_root)


        assert dev_tool.is_in_primitive(sample_weights, (None, 1)) or len(sample_weights) <= length, "Invalid weights"

        self._set_weights(sample_weights=sample_weights, index=index)

    def _set_weights(self, sample_weights, index=None):
        """Set the weights"""
        index = self._index if index is None else index
        length = len(self) if index is None else len(index)

        if dev_tool.is_in_primitive(sample_weights, (None, 1)):
            if index is None or len(self) == len(index):
                self._weights = 1
                return
            else:
                sample_weights = pd.Series(np.ones(len(index)), index=index)
        #    else:
        #        sample_weights = np.ones(length)
        else:
            sample_weights = pd.Series(sample_weights, index=index, dtype='f8')

        if len(self) == length and index is None:
            self._weights = sample_weights
        else:
            if dev_tool.is_in_primitive(self._weights, (None, 1)):
                self._weights = pd.Series(np.ones(len(self)), index=self._index)
            self._weights.update(sample_weights)

    def _scale_weights(self, index=None, weights_as_events=False, cast_int=True, min_weight=None):
        """Scale the weights to have minimum *weights_as_events* or min_weight"""
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        has_negative = False

        if dev_tool.is_in_primitive(self._weights, (None, 1)):
            weights = pd.Series(np.ones(length))
            min_w = min_neg_w = 1
        else:
            if index is None:
                weights = self._weights
            else:
                weights = self._weights.loc[index]
            # take care of negative weights
            min_w = min([w for w in weights if w > 0])
            min_neg_w = min(weights)

        if min_w != min_neg_w:
            for i, w in enumerate(weights):
                if w == 0:
                    w = -0.0001
                if w < 0:
                    has_negative = True
                    # rescale to [1, 0], mirror, multiply. Every neg weight is at
                    # least 0.5 as big as the min of positives
                    weights.iloc[i] = min_w * 0.5 * (1.0005 - float(w)/min_neg_w)

        if min_weight is not None:
            if min_weight == 0:
                min_weight = -0.0001
            elif min_weight < 0:
                has_negative = True

                min_weight = min_w * 0.2 * (1.0005 - float(min_weight)/min(min_neg_w, min_weight))

        min_weights = float(min(weights))


        if has_negative:
            self.logger.info("negative weights occured")

        weights = weights / min_weights * weights_as_events
        weights = np.array(map(round, weights))
        if cast_int:
            weights = weights + 0.00005
            weights = np.array(map(int, weights))
        assert min(weights) >= 0.98, "weights are not higher then 1, but they should be."
        return pd.Series(weights, index=index)

    def pandasDF(self, columns=None, index=None, weights_as_events=False, min_weight=None,
                 selection=None):
        """Convert the data to pandas or cut an already existing data frame and
        return it.

        Return a pandas DataFrame

        Parameters
        ---------
        columns : str
            Arguments for the :py:func:`~root_numpy.root2rec` ls
            function.
        index : 1-D list
            The index from the root-branche to be used. If None, all indices
            will be used (all the HEPDataStorage instance was created with).
        weights_as_events : boolean or int >= 1
            If False, the data will returned as usual. If an integer is
            provided, this has three effects:

            1. The **weights** are divided by it's smallest element (so
               the smallest weight will be one)

            2. multiplied by the provided integer

            3. each event is multiple times recreated and added to the data
               according to its (new created) weight.

            As the weight of an event basically is the occurence of it, this
            can be seen as a *weight-to-events* conversion and is useful,
            if you have few data with sometimes high weights (which then can
            cause statistic effects if badly distributed)

            Don't forget to manually assing weights 1 and not just take
            the weights from data.

        .. note:: With highly unbalanced weights this can lead to a memory
                  explosion!

        min_weight : int or float
            For *weights_as_events*; normaly, the minimum of the weights
            times *weights_as_events* is taken for scaling of the weights,
            if a *min_weight* is provided, the minimum of the weights and
            *min_weight* times the *weights_as_events* will be taken for
            scaling.

        selection : str
            A selection applied to **ROOT TTrees** only! For further details,
            see the root_numpy module.
        """

        # initialize variables
        index = self._index if index is None else list(index)
        columns = self.columns if columns is None else data_tools.to_list(columns)

        # flag whether we have to add new data or not (convert weights to events)
        convert_data = (weights_as_events >= 1) and not dev_tool.is_in_primitive(self.get_weights(
                                                        index=index), (None, 1))

        # create data
        data_out = self._make_df(columns=columns, index=index, selection=selection, copy=True)
        if not data_out.index.tolist() == range(len(data_out)):  # if not, convert the indices to
            data_out.reset_index(drop=True, inplace=True)

        # weights to number of events conversion
        if convert_data:
            # get new weights > weights_as_events OR > min_weight
            weights = self._scale_weights(index=index, weights_as_events=weights_as_events,
                                          cast_int=True, min_weight=min_weight)
            n_rows = sum(weights)
            starting_row = len(data_out)
            try:
                data_out = data_out.append(pd.DataFrame(index=range(starting_row, n_rows),
                                                    columns=data_out.columns))
            except MemoryError as error:
                meta_config.error_occured()
                self.logger.critical("Memory error occured during conversion of weights to events" +
                                     "\nMost propably you have very large/small weights" +
                                     "\nExtended problem report:" +
                                     "\nWeights_as_events = " + str(weights_as_events) +
                                     "\nmin_weight (parameter) = " + str(min_weight) +
                                     "\nSelf._weights: " +
                                     "\n  min = " + str(np.min(self._weights)) +
                                     "\n  max = " + str(np.max(self._weights)) +
                                     "\n  mean = " + str(np.mean(self._weights)) +
                                     "\n Number of total new events (the critical part!): " +
                                     str(n_rows)
                                     )
                raise error

            #test
            np_data = data_out.as_matrix()

            self.logger.info("Length of data was " + str(len(weights)) + ", new one will be " + str(sum(weights)))
            new_row = starting_row  # starting row for adding data == endrow of first dataframe

            for row_ori in xrange(starting_row):
                weight = weights.iloc[row_ori]
                if new_row %3000 == 0:
                    self.logger.info("adding row nr " + str(row_ori) + " with weight " + str(weight))
                if weight == 1:
                    continue  # there is no need to add extra data in this "special case"
                else:
                    for tmp_ in range(1, weight):
                        np_data[new_row] = np_data[row_ori]
                        #data_out.iloc[new_row] = data_out.iloc[row_ori]
                        new_row += 1
            self.logger.info("data_out Dataframe created")

            assert new_row == n_rows, "They should be the same in the end"
            #test
            data_out = pd.DataFrame(np_data)

        # reassign branch names after conversation.
        # And pandas has naming "problems" if only 1 branch
        data_out.columns = columns
        return data_out

    def _make_df(self, data=None, columns=None, index=None, copy=False, selection=None):
        """Return a DataFrame from the given data. Does some dirty, internal work."""
        # initialize data
        # TODO: remove trailing comment?
        data = self._data # if dev_tool.is_in_primitive(data) else data
        columns = self.columns if columns is None else data_tools.to_list(columns)
        index = self._index if index is None else data_tools.to_list(index)

        if self._data_type == 'root':
            #update root dictionary
            temp_root_dict = dict(data, **{'branches': columns, 'selection': selection})
            for key, val in temp_root_dict.items():
                if dev_tool.is_in_primitive(val, None):
                    temp_root_dict[key] = self.data.get(key)
            data = data_tools.to_pandas(temp_root_dict)

        elif self._data_type == 'array':
            data = pd.DataFrame(data, index=index, columns=columns, copy=copy)
        elif self._data_type == 'df':
            if columns is not None:
                data = data[columns]
        else:
            raise NotImplementedError("Unknown/not yet implemented data type")

        assert isinstance(data, pd.DataFrame), "data did not convert correctly"
        data = data if index is None else data.loc[index]

        return data

    def get_labels(self, columns=None, as_list=False):
        """Return the human readable branch-labels of the data.

        Parameters
        ----------
        columns : list with str or str
            The labels of the columns to return
        as_list : boolean
            If true, the labels will be returned as a list instead of a dict.

        Return
        ------
        out : list or dict
            Return a list or dict containing the labels.
        """
        if columns is None:
            columns = self.columns
        columns = data_tools.to_list(columns)
        if as_list:
            labels_out = [self._label_dic.get(col, col) for col in columns]
        else:
            labels_out = {key: self._label_dic.get(key) for key in columns}
        return labels_out

    def set_labels(self, data_labels, add_label=True):
        """Set the human readable data-labels (for the columns).

        Sometimes you want to change the labels of columns. This can be done
        by passing a dictionary containing the column as key and a
        human-readable name as value.

        Parameters
        ----------
        data_labels : dict
            It has the form: {column: name}
        add_label : boolean
            If False, the existing label dictionary gets overwritten instead
            of updated.
        """
        if data_labels is None:
            return
        assert isinstance(data_labels, dict), "Not a dictionary"
        self._set_data_labels(data_labels=data_labels, add_label=add_label)

    def _set_data_labels(self, data_labels, add_label=True):
        """Update the data labels"""

        if add_label:
            self._label_dic.update(data_labels)
        else:
            self._label_dic = data_labels

    def get_targets(self, index=None, weights_as_events=False, min_weight=None):
        """Return the targets of the data **as a numpy array**."""
        # assing defaults
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        # get targets via internal method
        out_targets = self._get_targets(index=index, weights_as_events=weights_as_events,
                                        min_weight=min_weight)

        # create targets if targets are "simpel" for output
        if dev_tool.is_in_primitive(out_targets, (-1, 0, 1, None)):
            if self._target is None:
                self.logger.warning("Target list consists of None!")
            out_targets = dev_tool.make_list_fill_var([], length, self._target)
            out_targets = pd.Series(out_targets, index=index)

        return out_targets

    def _get_targets(self, index=None, weights_as_events=False, min_weight=None):
        """Return targets as pandas Series, crashes the index if weights_as_events!"""
        # assign defaults
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        # If weights_as_events and the target is only primitiv, make targets
        if dev_tool.is_in_primitive(self._target, (-1, 0, 1, None)) and weights_as_events >= 1:
            if self._target is None:
                self.logger.warning("Target list consists of None!")
            out_targets = dev_tool.make_list_fill_var([], length, self._target)
        else:
            if index is None or dev_tool.is_in_primitive(self._target, (-1, 0, 1, None)):
                out_targets = self._target
            else:
                out_targets = self._target.loc[index]

        if weights_as_events >= 1:
            temp_target = deque()
            weights = self._scale_weights(index, weights_as_events=weights_as_events,
                                          cast_int=True, min_weight=min_weight)
            assert len(weights) == len(out_targets), "wrong length of weights and targets"
            for weight, target in itertools.izip(weights, out_targets):
                if weight == 1:
                    continue
                else:
                    weight -= 1  # one target is already contained in out_targets
                    temp_target.append([target] * weight)
            out_targets = out_targets.append(temp_target)
        out_targets = pd.Series(out_targets, index=index)

        return out_targets

    def set_targets(self, targets, index=None):
        """Set the targets of the data. Either a list-like object or
        {-1, 0, 1, None}"""

        if not dev_tool.is_in_primitive(targets, (-1, 0, 1, None)):
            assert len(self) == len(targets), "Invalid targets"
        self._set_target(target=targets, index=index)

    def _set_target(self, target, index=None):
        """Set the target. Attention with Series, index must be the same as data-index"""
        index = self._index if dev_tool.is_in_primitive(index) else index
        if isinstance(target, (list, np.ndarray, pd.Series)):
            target = pd.Series(target, index=index, copy=True)
            target.sort_index(inplace=True)
        self._target = target

    def make_dataset(self, second_storage=None, index=None, index_2=None, columns=None,
                     weights_as_events=False, weights_as_events_2=False,
                     targets_from_data=False,
                     min_weight=None, weights_ratio=0, shuffle=False):

        """Create data, targets and weights of the instance (and another one)

        In machine-learning, it is very often required to have data, it's
        targets (or labeling, the 'y') and the weights. In most cases, we
        are not only interested in one such pair, but need to concatenate
        it to other data (for example signal and background).

        This is exactly, what make_dataset does.

        Parameters
        ----------
        second_storage : instance of
        :py:class:`~raredecay.tools.data_storage.HEPDataStorage`
            A second data-storage. If provided, the data/targets/weights
            will be concatenated and returned as one.
        index : list(int, int, int, ...)
            The index for the **calling** (the *first*) storage instance.
        index_2 : list(int, int, int, ...)
            The index for the (optional) **second storage instance**.
        columns : list(str, str, str, ...)
            The columns to be used of **both** data-storages.
        weights_as_events : int
            Convert the weights to number of events. Argument specifies the
            minimum weight (if not provided by min_weights).
        min_weight : float
            To scale the weights to events, the weights get divided by the
            lowest weight. If a min_weights is provided *and* it is lower
            then the smallest weight, the min_weight will be used instead.

        ..note:: If weights_ratio is provided, this parameter is ignored.
        weights_ratio : float >= 0
            Only works if a second data storage is provided and assumes
            that the two storages can be seen as the two different targets.
            If zero, nothing happens. If it is bigger than zero, it
            represents the ratio of the sum of the weights from the first
            to the second storage. If set to 1, they both are equally
            weighted. Requires internal normalisation.

            Ratio := sum(weights1) / sum(weights2)
        shuffle : boolean
            If True, the dataset will be shuffled before returned
         """
         # TODO1: implement take_target_from_data
         # TODO2: make it recursive. second storage can be a list and get
         # data via make_dataset (or similar... difficult with ratio etc...)
        # initialize values
        if shuffle:
             warnings.warn("Shuffle not yet implemented!!")

        normalize_1 = 1
        normalize_2 = 1

        # set min_weight and get some later needed variables
        if second_storage is not None:
            weights_1 =self.get_weights(index=index)
            weights_2 = second_storage.get_weights(index=index_2)
            min_weight_1 = float(min(weights_1))
            min_weight_2 = float(min(weights_2))
            min_w1_w2 = min((min_weight_1, min_weight_2))
            if min_weight is not None:
                min_weight = min((min_w1_w2, min_weight))

        if weights_ratio > 0 and second_storage is not None:
            min_weight = None  # ignore if weigths_ratio > 0

            sum_weight_1 = float(sum(weights_1))
            sum_weight_2 = float(sum(weights_2))

            ratio_1 = weights_ratio * sum_weight_2 / sum_weight_1
            self.logger.info("ratio_1 = " + str(ratio_1))
            if ratio_1 >= 1:
                ratio_2 = 1.0
            else:
                ratio_2 = 1.0 / ratio_1
                ratio_1 = 1.0

            if weights_as_events in (False, None) and weights_as_events_2 in (False, None):
                normalize_1 = ratio_1
                normalize_2 = ratio_2

            elif weights_as_events is False and weights_as_events_2 > 0:
                ratio_1 /= min_weight_2

                min_ratio = min((ratio_1, ratio_2))
                ratio_1 /= min_ratio
                ratio_2 /= min_ratio

                scale_factor = weights_as_events_2 / ratio_2  # scale to weights_as_events

                normalize_1 = ratio_1 * scale_factor
                weights_as_events_2 = int(np.round(ratio_2 * scale_factor) + 0.00005)
                print "normalize_1", normalize_1

            elif weights_as_events > 0 and weights_as_events_2 is False:
                ratio_2 /= min_weight_1

                min_ratio = min((ratio_1, ratio_2))
                ratio_1 /= min_ratio
                ratio_2 /= min_ratio

                scale_factor = weights_as_events / ratio_1  # scale to weights_as_events

                weights_as_events = int(np.round(ratio_1 * scale_factor) + 0.00005)
                normalize_2 = ratio_2 * scale_factor
                print "normalize_2", normalize_2

            else:
                # equalize the additional factor from the weights_to_events (division by min(weight))
                ratio_1 /= min_weight_2
                ratio_2 /= min_weight_1

                min_ratio = min((ratio_1, ratio_2))
                ratio_1 /= min_ratio
                ratio_2 /= min_ratio

                max_converter = max((weights_as_events, weights_as_events_2))
                weights_as_events = int(np.round(ratio_1 * max_converter) + 0.00005)
                weights_as_events_2 = int(np.round(ratio_2 * max_converter) + 0.00005)


        data = self.pandasDF(columns=columns, index=index, min_weight=min_weight,
                             weights_as_events=weights_as_events)
        if second_storage is None:
            targets = self.get_targets(index=index, min_weight=min_weight,
                                       weights_as_events=weights_as_events)
        weights = self.get_weights(index=index, min_weight=min_weight,
                                   weights_as_events=weights_as_events, normalize=normalize_1)

        if second_storage is not None:
            assert isinstance(second_storage, HEPDataStorage), "Wrong type, has to be a HEPDataStorage"
            length_1 = len(data)
            data_2 = second_storage.pandasDF(columns=columns, index=index_2,
                                             min_weight=min_weight,
                                             weights_as_events=weights_as_events_2)
            length_2 = len(data_2)
            data = pd.concat((data, data_2), ignore_index=True, copy=False)

            if targets_from_data:
                if weights_as_events or weights_as_events_2:
                    warnings.warn("targets_from_data for weights_to_events not working yet! most probably...")
                targets_1 = self.get_targets()
                targets_2 = second_storage.get_targets()
                targets = np.concatenate(targets_1, targets_2)
            else:
                targets = np.concatenate((np.zeros(length_1), np.ones(length_2)))

            weights_2 = second_storage.get_weights(index=index_2, min_weight=min_weight,
                                                   weights_as_events=weights_as_events_2,
                                                   normalize=normalize_2)
            weights = np.concatenate((weights, weights_2))

        return data, targets, weights

    def copy_storage(self, columns=None, index=None):
        """Return a copy of self with only some of the columns (and therefore
        labels etc) or indices.

        Parameters
        ----------
        columns : str or list(str, str, str, ...)
            The columns which will be in the new storage.
        index : 1-D array-like
            The indices of the rows (and corresponding weights, targets etc.)
            for the new storage.
        """
        index = self._index if index is None else list(index)
        columns = self.columns if columns is None else columns

        new_labels = self.get_labels(columns=columns)
        new_data = copy.deepcopy(self.data)

        new_targets = copy.deepcopy(self._get_targets(index=index))
        new_weights = copy.deepcopy(self._get_weights(index=index))
        new_index = copy.deepcopy(index)


        new_storage = HEPDataStorage(new_data, target=new_targets,
                                     sample_weights=new_weights,
                                     data_labels=new_labels, index=new_index,
                                     data_name=self.data_name,
                                     data_name_addition=self.data_name_addition + " cp")
        new_storage.columns = columns
        return new_storage

    def get_LabeledDataStorage(self, columns=None, index=None, random_state=None, shuffle=False):
        """Create and return an instance of class "LabeledDataStorage" from
        the REP repository.

        Return
        ------
        out: LabeledDataStorage instance
            Return a Labeled Data Storage instance created with the data
            from inside this instance.
        """
        index = self.index if index is None else list(index)
        columns = self.columns if columns is None else columns
        new_lds = LabeledDataStorage(self.pandasDF(columns=columns, index=index),
                                     target=self.get_targets(index=index),
                                     sample_weight=self.get_weights(index=index),
                                     random_state=random_state, shuffle=shuffle)
        return new_lds

    def make_folds(self, n_folds=10, shuffle=True):
        """Create shuffled train-test folds which can be accessed via
        :py:meth:`~raredecay.tools.data_storage.HEPDataStorage.get_fold()`

        Split the data into n folds (for usage in KFold validaten etc.).
        Then every fold consists of a train dataset, which consists of
        n-1/n part of the data and a test dataset, which consists of 1/n part
        of the whole data.
        The folds will be created as *HEPDataStorage*.
        To get a certain fold (train-test pair), use
        :py:meth:`~raredecay.tools.data_storage.HEPDataStorage.get_fold()`

        Parameters
        ----------
        n_folds : int > 1
            The number of folds to be created from the data. If you want, for
            example, a simple 2/3-1/3 split, just specify n_folds = 3 and
            just take one fold.
        """
        if n_folds <= 1:
            raise ValueError("Number of folds has to be higher then 1")

        self._fold_index = []

        # split indices of shuffled list
        length = len(self)
        temp_indeces = [int(round(length/n_folds)) * i for i in range(n_folds)]
        temp_indeces.append(length)  # add last index. len(index) = n_folds + 1

        # get a copy of index and shuffle it if True
        temp_index = copy.deepcopy(self._make_index())
        if shuffle:
            random.shuffle(temp_index)
        for i in range(n_folds):
            self._fold_index.append(temp_index[temp_indeces[i]:temp_indeces[i + 1]])

    def get_fold(self, fold):
        """Return the specified fold: train and test data as instance of
        :py:class:`~raredecay.tools.data_storage.HEPDataStorage`

        Parameters
        ----------
        fold : int
            The number of the fold to return. From 0 to n_folds - 1
        Return
        ------
        out : tuple(HEPDataStorage, HEPDataStorage)
            Return the *train* and the *test* data in a HEPDataStorage
        """
        assert self._fold_index is not None, "Tried to get a fold but data has no folds. First create them (make_folds())"
        assert isinstance(fold, int) and fold<len(self._fold_index), "your value of fold is not valid"
        train_index = []
        for i, index_slice in enumerate(self._fold_index):
            if i == fold:
                test_index = copy.deepcopy(index_slice)
            else:
                train_index += copy.deepcopy(index_slice)
        n_folds = len(self._fold_index)
        test_DS = self.copy_storage(index=test_index)
        test_DS._fold_status = (fold, n_folds)
        test_DS.fold_name = "test set fold " + str(fold + 1) + " of " + str(n_folds)  # + 1 human-readable
        train_DS = self.copy_storage(index=train_index)
        train_DS._fold_status = (fold, n_folds)
        train_DS.fold_name = "train set fold " + str(fold + 1) + " of " + str(n_folds)
        return train_DS, test_DS


    def get_n_folds(self):
        """Return how many folds are currently availabe or 0 if no folds
        have been created

        Return
        ------
        out : int
            The number of folds which are currently available.
        """
        return 0 if self._fold_index is None else len(self._fold_index)

    def plot_correlation(self, second_storage=None, columns=None,
                         method='pearson', plot_importance=5):
        """Plot the feature correlation for the data (combined with other data)

        Calculate the feature correlation, return it and plot them.

        Parameters
        ----------
        second_storage : HEPDataStorage or None
            If a second data-storage is provided, the data will be merged and
            then the correlation will be calculated. Otherwise, only this
            datas correlation will be calculated and plotted.
        method : str {'pearson', 'kendall', 'spearman'}
            The method to calculate the correlation.
        plot_importance : int {1, 2, 3, 4, 5}
            The higher the more likely it gets plotted. Depends on the
            plot_verbosity. To make sure the correlation...
            - *does* get plotted, chose 5
            - does *not* get plotted, chose 1

        Return
        ------
        out : pandas DataFrame
            Return the feature-correlations in a pandas DataFrame
        """
        data, _tmp, _tmp2 = self.make_dataset(second_storage=second_storage,
                                              shuffle=True, columns=columns)
        del _tmp, _tmp2
        correlation = data.corr(method=method)
        corr_plot = sns.heatmap(correlation.T)

        # turn the axis label
        for item in corr_plot.get_yticklabels():
            item.set_rotation(0)

        for item in corr_plot.get_xticklabels():
            item.set_rotation(90)


    def plot(self, figure=None, title=None, data_name=None, std_save=True,
             log_y_axes=False, columns=None, index=None, sample_weights=None,
             data_labels=None, see_all=False, hist_settings=None, weights_as_events=False):
        """Draw histograms of the data.

        .. warning:: Only 99.98% of the newest plotted data will be shown to focus
           on the essential parts (the axis limits will be set accordingly).
           This implies a risk of cutting the previously (in the same figure)
           plotted data (mostly, if they do not overlap a lot). To ensure that
           all data is plotted, set *see_all* to *True*.

        Parameters
        ----------
        figure : str or int
            The name of the figure. If the figure already exists, the plots
            will be plotted in the same window (can be intentional, for
            example to compare data)
        title : str
            | The title of the whole plot (NOT of the subplots). If several
              titles for the same figures are given, they will be *concatenated*.
            | So for a "simple" title, specify the title only once.
        data_name:
            | Additional, (to the *data_name* and *data_name_addition*), human-
              readable name for the legend.
            | Examples: "before cut", "after cut" etc
        std_save : boolean
            If True, the figure will be saved (with
            :py:meth:`~raredecay.tools.output.output.save_fig()`) with
            "standard" parameters as specified in *meta_config*.
        log_y_axes : boolean
            If True, the y-axes will be scaled logarithmically.
        columns : str or list(str, str, str, ...)
            The columns of the data to be plotted. If None, all are plotted.
        index : list(int, int, int, ...)
            A list of indeces to be plotted. If None, all are plotted.
        sample_weights : list containing weights
            The weights for the data, how "high" a bin is. Actually, how much
            it should account for the whole distribution or how "often" it
            occures.
        data_labels : dict
            Contain the column as key and the value as label
        hist_settings : dict
            A dictionary containing the settings as keywords for the
            :py:func:`~matplotlib.pyplot.hist()` function.

        """
#==============================================================================
#        initialize values
#==============================================================================
        # update labels
        if dev_tool.is_in_primitive(data_labels, None):
            data_labels = {}
        data_labels = dict(self._label_dic, **data_labels)

        # update weights
        if dev_tool.is_in_primitive(sample_weights, None):
            if weights_as_events in (False, None):
                sample_weights = self.get_weights(index=index)
        #assert ((len(sample_weights) == (len(self.get_weights(index=index))) or
        #            dev_tool.is_in_primitive(sample_weights, None)) ,
        #            "sample_weights is not the right lengths")

        # update hist_settings
        if dev_tool.is_in_primitive(hist_settings, None):
            hist_settings = {}
        if isinstance(hist_settings, dict):
            hist_settings = dict(meta_config.DEFAULT_HIST_SETTINGS, **hist_settings)

        # create data
        data_plot = self.pandasDF(columns=columns, index=index, weights_as_events=weights_as_events)
        columns = data_plot.columns.values
        self.logger.debug("plot columns from pandasDataFrame: " + str(columns))
        if weights_as_events >= 1:
            sample_weights=None

        # set the right number of rows and columns for the subplot
        subplot_col = int(math.ceil(math.sqrt(len(columns))))
        subplot_row = int(math.ceil(float(len(columns))/subplot_col))

        # assign a free figure if argument is None
        if dev_tool.is_in_primitive(figure, None):
            while True:
                safety = 0
                figure = self.__figure_number + 1
                self.__figure_number += 1
                assert safety < meta_config.MAX_FIGURES, "stuck in an endless while loop"
                if figure not in self.__figure_dic.keys():
                    x_limits_col = {}
                    self.__figure_dic.update({figure: x_limits_col, 'title': ""})
                    break
        elif figure not in self.__figure_dic.keys():
            x_limits_col = {}
            self.__figure_dic.update({figure: x_limits_col, 'title': ""})
        out_figure = plt.figure(figure, figsize=(20, 30))

        # create a label
        label_name = data_tools.obj_to_string([self._name[0], self._name[1],
                                               data_name], separator=" - ")
        self.__figure_dic['title'] += "" if title is None else title
        plt.suptitle(self.__figure_dic.get('title'), fontsize=self.supertitle_fontsize)

#==============================================================================
#       Start plotting
#==============================================================================
        # plot the distribution column by column
        for col_id, column in enumerate(columns, 1):
            # only plot in range x_limits, otherwise the plot is too big
            x_limits = self.__figure_dic.get(figure).get(column, None)
            lower, upper = np.percentile(np.hstack(data_plot[column]),
                                         [0.01, 99.99])
            if dev_tool.is_in_primitive(x_limits, None):
                x_limits = (lower, upper)
            elif see_all:  # choose the maximum range. Bins not nicely overlapping.
                x_limits = (min(x_limits[0], lower), max(x_limits[1], upper))
            self.__figure_dic[figure].update({column: x_limits})
            plt.subplot(subplot_row, subplot_col, col_id)
            temp1, temp2, patches = plt.hist(data_plot[column], weights=sample_weights,
                                             log=log_y_axes, range=x_limits,
                                             label=label_name, **hist_settings)
            plt.title(column)
        plt.legend()

        if std_save and out_imported:
            out.save_fig(out_figure, **meta_config.DEFAULT_SAVE_FIG)
        return out_figure

    def plot2Dscatter(self, x_branch, y_branch, dot_size=20, color='b', weights=None, figure=0):
        """Plots two columns against each other to see the distribution.

        The dots size is proportional to the weights, so you have a good
        overview on the data and the weights.

        Parameters
        ----------
        x_branch : str

        """
        # TODO: make nice again
        out_figure = plt.figure(figure)
        if isinstance(weights, (int, long, float)):
            weights = dev_tool.make_list_fill_var(weights, length=len(self),
                                                  var=weights)
        else:
            weights = self.get_weights()
        assert len(weights) == len(self), "Wrong length of weigths"
        size = [dot_size*weight for weight in weights]
        temp_label = data_tools.obj_to_string([i for i in self._name])
        plt.scatter(self.pandasDF(columns=x_branch),
                    self.pandasDF(columns=y_branch), s=size, c=color,
                    alpha=0.5, label=temp_label)
        plt.xlabel(self.get_labels(columns=x_branch, as_list=True))
        plt.ylabel(self.get_labels(columns=y_branch, as_list=True))
        plt.legend()

        return out_figure

# TODO: add correlation matrix

if __name__ == '__main__':

    n_tested = 0

    b = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43], [51, 52, 53]])
    a = pd.DataFrame(b, columns=['one', 'two', 'three'], index=[1,2,11,22,33])
    c = copy.deepcopy(a)
    storage = HEPDataStorage(a, index=[1,2,11,22], target=[1,1,1,0], sample_weights=[1,2,1,0.5],
                             data_name="my_data", data_name_addition="and addition")
    n_tested += 1

    d = a.loc[[1,2,11,22]]
    pd1 = storage.pandasDF()
    pd2 = storage.pandasDF(weights_as_events=2)
    scaled_w = storage.get_weights(weights_as_events=2)

    t1 = all(pd1 == d.reset_index(drop=True))
    t2 = len(pd2) == 18 and all(scaled_w == np.ones(len(pd2)))
    t3 = all(storage.get_targets(index=[1, 11, 22]) == np.array([1, 1, 0]))
    works1 = t1 and t2 and t3

    print "storage with DataFrame works:", works1

    DATA_PATH = '/home/mayou/Big_data/Uni/decay-data/'
    all_branches = ['B_PT', 'nTracks', 'nSPDHits'
              , 'B_FDCHI2_OWNPV', 'B_DIRA_OWNPV'
              ,'B_IPCHI2_OWNPV', 'l1_PT', 'l1_IPCHI2_OWNPV','B_ENDVERTEX_CHI2',
              'h1_IPCHI2_OWNPV', 'h1_PT', 'h1_TRACK_TCHI2NDOF'
              ]
    cut_Bu2K1Jpsi_mc = dict(
        filenames=DATA_PATH+'cut_data/CUT-Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
        treename='DecayTree',
        branches=all_branches

    )

    root_data = dict(
    data=cut_Bu2K1Jpsi_mc,
    sample_weights=None,
    data_name="B->K1 J/Psi monte-carlo",
    data_name_addition="cut"
    )

    storage2 = HEPDataStorage(**root_data)

    storage2.plot_correlation()

    storage3 = storage2.copy_storage(index=[3,5,7,9], columns=['B_PT', 'nTracks'])
    df11 = storage3.pandasDF()
    df12 =  storage3.pandasDF(weights_as_events=3, min_weight=4)
    storage3.set_weights(sample_weights=[1,4,1,0.5])
    #print "df11", df11
    #print "df12", df12
    #print storage3.pandasDF(weights_as_events=3, min_weight=4)
    w3 = storage3.get_weights(weights_as_events=3)

    storage3.make_folds(4)
    train, test = storage3.get_fold(1)
    print train.pandasDF(), "\n", test.pandasDF()
    train.make_folds(3)
    train1, test1 = train.get_fold(1)
    print train1.pandasDF(), "\n", test1.pandasDF()
    t21 = isinstance(storage2.pandasDF(), pd.DataFrame)
    print "t21 = ", t21
    #print storage2.pandasDF().index.values
    #print "w3 = ", w3, "type:", type(w3)



    t22 = all(w3 == np.ones(39))
    #print "t22 =", t22
    works2 = t21 and t22
    works = works1 and works2
    print "DataFrame works:", works
    plt.show()



    print "Selftest finished, tested " + str(n_tested) + " functions."





