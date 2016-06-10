# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:10:29 2016

@author: mayou
"""
# debug
from __future__ import division, absolute_import

import copy
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import deque

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

# TODO: import config not needed??
# import configuration
#import importlib
#from raredecay import meta_config
#cfg = importlib.import_module(meta_config.run_config)
modul_logger = dev_tool.make_logger(__name__, **meta_config.DEFAULT_LOGGER_CFG)


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
        self.logger = modul_logger if logger is None else logger

        # initialize data
        self._fold_index = None  # list with indeces of folds
        self._fold_status = None  # tuple (my_fold_number, total_n_folds)
        self.set_data(data=data)

        # data name
        self._name = ["", "", ""]
        data_name = "unnamed data" if data_name is None else data_name
        self.data_name = data_name
        self.data_name_addition = data_name_addition
        self.fold_name = None

        # initialize targets
        self._set_target(target=target)

        # data-labels human readable, initialize with the column name
        self._label_dic = {col: col for col in self.columns if self._label_dic.get(col) is None}
        self.set_labels(data_labels=data_labels, add_label=add_label)

        # initialize weights
        self._set_weights(sample_weights)

        # plot settings
        if dev_tool.is_in_primitive(hist_settings, None):
            hist_settings = meta_config.DEFAULT_HIST_SETTINGS
        self.hist_settings = hist_settings
        self.supertitle_fontsize = supertitle_fontsize

    @property
    def __len__(self):
        return self._length

# TODO: remove obsolet
    def get_name(self):
        """Return the human-readable name of the data as a string"""
        warnings.warn(DeprecationWarning, "Depreceated, will be removed. Use obj.name instead.)
        return self._get_name()

    @property
    def name(self):
        """Return the **full** human-readable name of the data as a string"""
        return self._get_name()

    def _get_name(self):
        out_str = data_tools.obj_to_string(self._name, separator=" ")
        return out_str

    def _set_name(self, data_name, data_name_addition, fold_name):
        """Set the data name"""
        # initialize name
        if self._name is None:

        # set the new name in self._name
        for i, name in enumerate([data_name, data_name_addition, fold_name]):
            if name is not None:
                self._name[i] = name

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
        warnings.warn(FutureWarning, "Will be removed in the future. Use obj.index instead")
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
            if self._data_type == 'root':
                pass  # no index contained in root-dicts
            elif self._data_type == 'array':
                pass  # no index information contained in an array
            elif self._data_type == 'df':
                index_list = self._data.index.tolist()
                if not  index_list == range(len(self)):  # if special indexing
                    self._index = index_list
        else:
            self._index = index

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        if columns is None:
            self._set_columns
        else:
            # TODO: maybe check?
            self._columns = columns

    def _set_columns(self):
        if self._data_type == 'root':
            self._columns = data_tools.to_list(self._data['branches'])
        elif self._data_type == 'df':
            self._columns = data_tools.to_list(self._data.columns.values)
        elif self._data_type == 'array':
            self._columns = ['feature_' + str(i) for i in range(len(self._data))]

    def _set_length(self):
         # determine whether to set length individually from the data or not
        if self._index is None:

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
            self._length = len(self._index)

    def _get_data_type(self, data):
        """Return the type of the data

        - 'df' : pandas DataFrame
        - 'root': root-file
        - 'array': numpy array
        """
        data_type = None
        if isinstance(data, dict):
            if data.has_key('filenames') and data['filenames'].endswith(self.__ROOT_DATATYPE):
                data_type = 'root'
        elif isinstance(data, pd.DataFrame):
            data_type = 'df'
        elif isinstance(data, (np.ndarray, np.array)):
            data_type = 'array'

        return data_type

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

        self.index = index
        self.columns = columns
        self._set_length()

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

    def get_rootdict(self, return_index=False):
        """Return the root-dictionary if available, else None"""
        warnings.warn(FutureWarning, "will be removed. Use obj.data instead")
        if return_index:
            return self._root_dict, self._index
        else:
            return self._root_dict




############# STOPPED WORKING HERE ######################################################################3



    def get_weights(self, normalize=True, index=None, weights_as_events=False, min_weight=None, **kwargs):
        """Return the weights of the specified indeces or, if None, return all.

        Parameters
        ----------
        normalize : boolean
            If True, the weights will be normalized to 1 (the average is 1).
            No effect, if weights_as_events is True.
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
            Return the weights in an array
        """
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        if dev_tool.is_in_primitive(self._weights, (None, 1)):
            normalize = False
            if kwargs.get('inter', False):  # intern use
                weights_out = self._weights
            else:
                weights_out = dev_tool.fill_list_var([], length, 1)
                weights_out = data_tools.to_ndarray(weights_out)
        elif weights_as_events >= 1:
            pass
        elif index is None:
            weights_out = data_tools.to_ndarray(self._weights)
        else:
            weights_out = data_tools.to_ndarray(self._weights[index])

        if (not dev_tool.is_in_primitive(self._weights, (None, 1))) and (weights_as_events >= 1):
            length = sum(self._scale_weights(index, weights_as_events=weights_as_events,
                                             cast_int=True, min_weight=min_weight))
            weights_out = np.ones(length)


        elif normalize:
            eps = 0.00001
            counter = 0
            while not (1-eps < weights_out.mean() and weights_out.mean() < 1 + eps):
                weights_out *= weights_out.size/weights_out.sum()
                counter += 1
                if counter > 100:  # to prevent endless loops
                    self.logger.warning("Could not normalize weights. Mean(weights): " + str(np.mean(weights_out)))
                    break
        return weights_out

    def set_weights(self, sample_weights, index=None, concat=False):
        """Set the weights of the sample.

        Parameters
        ----------
        sample_weights : 1-D array or list or int {1}
            The new weights for the dataset.
        index : 1-D array or list or None
            The indeces for the weights to be set
        concat : boolean
            If True, the weights will be concatenated to the already existing.
            Any conflict will be resolved in favour of the new ones.
        """
        index = self._index if index is None else index
        length = len(self) if index is None else len(index)
        assert dev_tool.is_in_primitive(sample_weights, (None, 1)) or len(sample_weights) == length, "Invalid weights"

        if dev_tool.is_in_primitive(sample_weights, (None, 1)):
            sample_weights = np.ones(length)
        if dev_tool.is_in_primitive(self._weights, (None, 1)) and concat:
            if self._index is None:
                self._weights = pd.Series(np.ones(len(self)))
            else:
                self._weights = pd.Series(np.ones(len(self)), index=self._index)
        sample_weights = pd.Series(sample_weights, index=index)
        if concat:
            self._weights.update(sample_weights)
        else:
            self._weights = sample_weights

    def _set_weights(self, sample_weights):
        """Set the weights"""

        if not dev_tool.is_in_primitive(sample_weights, (None, 1)):
            #assert len(sample_weights) == self._length
            sample_weights = pd.Series(sample_weights, index=self._index, copy=True)
        self._weights = sample_weights


    def _scale_weights(self, index, weights_as_events=False, cast_int=True, min_weight=None):
        """Scale the weights to have minimum *weights_as_events* or min_weight"""
        weights = self.get_weights(index=index)

        # take care of negative weights
        min_w = min([w for w in weights if w > 0])
        min_neg_w = min(weights)
        has_negative = False
        if min_w != min_neg_w:
            for i, w in enumerate(weights):
                if w == 0:
                    w = -0.0001
                if w < 0:
                    has_negative = True
                    # rescale to [1, 0], mirror, multiply. Every neg weight is at
                    # least 0.2 as big as the min of positives
                    weights.iloc[i] = min_w * 0.2 * (1.0005 - float(w)/min_neg_w)

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
        return weights

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
                                                        index=index, inter=True), (None, 1))

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
            data_out = data_out.append(pd.DataFrame(index=range(starting_row, n_rows), columns=data_out.columns))

            self.logger.info("Length of data was " + str(len(weights)) + ", new one will be " + str(sum(weights)))
            new_row = starting_row  # starting row for adding data == endrow of first dataframe
            for row_ori in xrange(starting_row):
                weight = weights[row_ori]
                if new_row %3000 == 0:
                    self.logger.info("adding row nr " + str(row_ori) + " with weight " + str(weight))
                if  weight == 1:
                    continue  # there is no need to add extra data in this "special case"
                else:
                    for tmp_ in xrange(1, weight):
                        data_out.iloc[new_row] = data_out.iloc[row_ori]
                        new_row += 1
            self.logger.info("data_out Dataframe created")

            assert new_row == n_rows, "They should be the same in the end"
        # reassign branch names after conversation.
        # And pandas has naming "problems" if only 1 branch
        data_out.columns = columns
        return data_out

    def _make_df(self, data=None, columns=None, index=None, copy=False, selection=None):
        """Return a DataFrame from the given data. Does some dirty, internal work."""
        # initialize data
        data = self._data if dev_tool.is_in_primitive(data) else data
        columns = self.columns if columns is None else data_tools.to_list(columns)
        index = self.index if index is None else data_tools.to_list(index)

        if self._data_type == 'root':
            #update root dictionary
            temp_root_dict = dict(data, **{'branches': columns, 'selection': selection})
            for key, val in temp_root_dict.items():
                if dev_tool.is_in_primitive(val, None):
                    temp_root_dict[key] = self._root_dict.get(key)
            data = data_tools.to_pandas(data)

        elif self._data_type == 'array':
            data = pd.DataFrame(data, index=index, columns=columns, copy=copy)

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
            columns = self._root_dict.get('branches')
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

        self.add_label = add_label
        self._label_dic.update(data_labels)





    def get_targets(self, index=None, weights_as_events=False, min_weight=None, **kwargs):
        """Return the targets of the data **as a numpy array**."""

        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        if dev_tool.is_in_primitive(self._target_label, (-1, 0, 1, None)):
            if kwargs.get('inter', False):
                return copy.deepcopy(self._target_label)
            else:
                if self._target_label is None:
                    self.logger.warning("Target list consists of None")
                out_targets = dev_tool.make_list_fill_var([], length, self._target_label)
        else:
            if index is None:
                out_targets = self._target_label
            else:
                out_targets = self._target_label[index]

        if weights_as_events >= 1:
            temp_target = deque()
            weights = self._scale_weights(index, weights_as_events=weights_as_events, cast_int=True, min_weight=min_weight)
            assert len(weights) == len(out_targets), "wrong length of weights and targets"
            for weight, target in weights, out_targets:
                if weight == 1:
                    continue
                else:
                    weight -= 1  # one target is already contained in out_targets
                    temp_target.append([target] * weight)
            out_targets = out_targets.append(temp_target)

        return np.array(out_targets)

    def set_targets(self, targets):
        """Set the targets of the data. Either a list-like object or
        {-1, 0, 1, None}"""

        if not dev_tool.is_in_primitive(targets, (-1, 0, 1, None)):
            assert len(self) == len(targets), "Invalid targets"
        self._set_target(targets)

    def _set_target(self, target):
        """Set the target"""
        if isinstance(target, (list, np.ndarray, pd.Series)):
            target = pd.Series(target, index=self._index, copy=True)
        self._target_label = target

    def copy_storage(self, columns=None, index=None):
        """Return a copy of self with only some of the columns (and therefore \
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
        new_labels = {}
        new_root_dic = copy.deepcopy(self._root_dict)

        if columns is not None:
            columns = data_tools.to_list(columns)
            new_root_dic['branches'] = copy.deepcopy(columns)

        new_targets = copy.deepcopy(self.get_targets(index=index, inter=True))
        new_weights = copy.deepcopy(self.get_weights(index=index, inter=True))
        new_index = copy.deepcopy(index)
        new_add_label = copy.deepcopy(self.add_label)

        for column in new_root_dic['branches']:
            new_labels[column] = copy.deepcopy(self._label_dic.get(column))

        new_storage = HEPDataStorage(new_root_dic, target=new_targets,
                                     sample_weights=new_weights,
                                     data_labels=new_labels, index=new_index,
                                     add_label=new_add_label, data_name=self._name[0],
                                     data_name_addition=self._name[1] + " cp")

        return new_storage

    def get_LabeledDataStorage(self, columns=None, index=None, random_state=None, shuffle=False):
        """Create and return an instance of class "LabeledDataStorage" from
        the REP repository.

        Return
        ------
        out: LabeledDataStorage instance
            Return a Labeled Data Storage instance created with the data
        """
        index = self._index if index is None else list(index)
        new_lds = LabeledDataStorage(self.pandasDF(columns=columns, index=index),
                                     target=self.get_targets(index=index),
                                     sample_weight=self.get_weights(index=index),
                                     random_state=random_state, shuffle=shuffle)
        return new_lds


    def make_folds(self, n_folds=10):
        """Create train-test folds which can be accessed via
        :py:meth:`~raredecay.tools.data_storage.HEPDataStorage.get_fold()`

        Parameters
        ----------
        n_folds : int > 1
            The number of folds to be created from the data. If you want, for
            example, a simple 2/3-1/3 split, just specify n_folds = 3 and
            just take one fold.
        """
        if n_folds <= 1:
            self.logger.error("Wrong number of folds. Set to default 10")
            n_folds = 10
            meta_config.error_occured()

        self._fold_index = []

        # split indices of shuffled list
        length = len(self)
        temp_indeces = [int(round(length/n_folds)) * i for i in range(n_folds)]
        temp_indeces.append(length)  # add last index. len(index) = n_folds + 1

        # get a copy of index and shuffle it
        temp_shuffled = copy.deepcopy(self._make_index())
        random.shuffle(temp_shuffled)
        for i in range(n_folds):
            self._fold_index.append(temp_shuffled[temp_indeces[i]:temp_indeces[i + 1]])

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
        test_DS._fold_name = "test set fold " + str(fold) + " of " + str(n_folds)
        train_DS = self.copy_storage(index=train_index)
        train_DS._fold_status = (fold, n_folds)
        train_DS._fold_name = "train set fold " + str(fold) + " of " + str(n_folds)
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
                    self.__figure_dic.update({figure: x_limits_col})
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
    works = t1 and t2
    print "DataFrame works:", works




    print "Selftest finished, tested " + str(n_tested) + " functions."





