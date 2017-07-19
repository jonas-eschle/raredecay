# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:10:29 2016

@author: Jonas Eschle "Mayou36"

This module contains the data handling. The main part is the class which
takes data, weights, targets, names and converts automatically, plots and more.
"""
from __future__ import division, absolute_import

import copy
import warnings
import math
import random

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from rep.data.storage import LabeledDataStorage

from raredecay.tools import data_tools, dev_tool
try:
    from raredecay.globals_ import out
    out_imported = True
except ImportError:
    warnings.warn(ImportWarning, "could not import out. Some functions regarding output" +
                  "(save figure etc.) won't be available")
    out_imported = False
from raredecay import meta_config

# TODO: import config not needed?? remove because its from the old structure
# import configuration
import importlib
# from raredecay import meta_config
cfg = importlib.import_module(meta_config.run_config)
modul_logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


class HEPDataStorage(object):
    """Data-storage for data, weights, targets; conversion; plots and more"""

    # define constants for independence
    __ROOT_DATATYPE = meta_config.ROOT_DATATYPE

    __figure_number = 0
    __figure_dic = {}
    latex_replacements = {
#        "CHI2": r"\chi^2",
        r"_PT": r" p_T",
        r"JPsi": r"J/\psi",
        r"K1": r"K_1 ",
        r"_1270": "",
        r"_ENDVERTEX_CHI2": r"\ \chi^2_{VTX}",
        r"_IPCHI2": r"\ \chi^2_{IP}",
        r"_FDCHI2": r"\ \chi^2_{FD}",
        r"_TRACK_CHI2": r"\ \chi^2_{track}",
        r"_OWNPV": r"\ ownPV",
        r"_CosTheta": r"\ cos(\theta)",
        r"NDOF": r"/N_{degree of freedom}",
        r"AMAXDOCA": r"\ AMAXDOCA",

#        "_": "\ "
        }

    def __init__(self, data, index=None, target=None, sample_weights=None,
                 data_name=None, data_name_addition=None, column_alias=None):
        """Initialize instance and load data.

        Parameters
        ----------
        data : |data_type|
            The data itself. This can be two different types

            - **root-tree dict** (*root-dict*):
              Dictionary which specifies all the information to convert a root-
              tree to an array. Directly given to :py:func:`~root_numpy.root2rec`
            - **pandas DataFrame**:
              A pandas DataFrame. The index (if not explicitly defined)
              and column names will be taken.

        index : 1-D array-like
            The indices of the data that will be used.
        target : list or 1-D array or int {0, 1}
            Labels the data for the machine learning. Usually the y.
        sample_weights : |sample_weights_type|
            |sample_weights_docstring|
            .. note:: If None or 1 specified, 1 will be assumed for all.
        data_name : str
            | Name of the data, human-readable. Displayed in the title of
              plots.
            | *Example: 'Bu2K1piee mc', 'beta-decay real data' etc.*
        data_name_addition : str
            | Additional remarks to the data, human readable. Displayed in
              the title of plots.
            | *Example: 'reweighted', 'shuffled', '5 GeV cut applied' etc.*
        column_alias : |column_alias_type|
            |column_alias_docstring|
        """
        # initialize logger
        self.logger = modul_logger

        # initialize index
        # self._index = None

        # initialize data
        # self._data = None
        self._data_type = None
        self.column_alias = {} if column_alias is None else column_alias
        self._fold_index = None  # list with indeces of folds
        self._fold_status = None  # tuple (my_fold_number, total_n_folds)
        self._length = None
        self.set_data(data=data, index=index)
        # self._columns = None

        # data name
        self._name = ["", "", ""]
        data_name = "unnamed data" if data_name is None else data_name
        self.data_name = data_name
        self.data_name_addition = data_name_addition
        self.fold_name = None

        # initialize targets
        self._set_target(target=target)

#        # data-labels human readable, initialize with the column name
#        self._label_dic = {}
#        self._label_dic = {col: col for col in self.columns if self._label_dic.get(col) is None}
        # TODO: delete?
#        self.set_labels(data_labels=data_labels)

        # initialize weights
        self._weights = None
        self.set_weights(sample_weights)

        # plot settings
        hist_settings = meta_config.DEFAULT_HIST_SETTINGS
        self.hist_settings = hist_settings
        self.supertitle_fontsize = 18

    def __len__(self):
        if self._length is None:
            self._set_length()
        return self._length

# TODO: remove obsolet
    def get_name(self):
        """Return the human-readable name of the data as a string."""
        warnings.warn("Depreceated, obj.get_name() will be removed. Use obj.name instead.", FutureWarning)
        return self._get_name()

    @property
    def name(self):
        """Return the **full** human-readable name of the data as a string."""
        return self._get_name()

    def _get_name(self):
        out_str = data_tools.obj_to_string(self._name, separator=" ")
        return out_str

    def _set_name(self, data_name=None, data_name_addition=None, fold_name=None):
        """Set the data name."""
        # set the new name in self._name
        for i, name in enumerate([data_name, data_name_addition, fold_name]):
            if name is not None:
                self._name[i] = name

# TODO: change the naming into a dict?
    @property
    def data_name(self):
        """The name of the data."""
        return self._name[0]

    @property
    def data_name_addition(self):
        """The data name addition."""
        return self._name[1]

    @property
    def fold_name(self):
        """The name of the fold (like *fold 2 of 5*)."""
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

    @property
    def data_type(self):
        """"Return the data-type like 'root', 'df' etc."""
        return self._data_type

    def get_index(self):
        """Return the index used inside the DataStorage. Advanced feature."""
        warnings.warn("Will be removed in the future. Use obj.index instead", FutureWarning)
        return self._make_index()

    @property
    def index(self):
        """The internal index"""
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
        """The columns/branches of the data"""
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

    def _set_length(self):
        # determine whether to set length individually from the data or not
        index = self._index
        if index is None:
            if self._data_type == 'root':
                temp_root_dict = copy.deepcopy(self._data)
                temp_branch = temp_root_dict.pop('branches')  # remove to only use one branch
                temp_branch = data_tools.to_list(temp_branch)
                self._length = len(data_tools.to_ndarray(dict(branches=temp_branch[0],
                                                              **temp_root_dict)))
            elif self._data_type == 'df':
                self._length = len(self._data)
            elif self._data_type == 'array':
                self._length = self._data.shape[1]

        else:
            self._length = len(index)

    @staticmethod
    def _get_data_type(data):
        """Return the type of the data.

        - 'df' : pandas DataFrame
        - 'root': root-file
        - 'array': numpy array
        """
        data_type = None
        if isinstance(data, dict):
            if 'filenames' in data and data['filenames'].endswith(HEPDataStorage.__ROOT_DATATYPE):
                data_type = 'root'
        elif isinstance(data, pd.DataFrame):
            data_type = 'df'
        elif isinstance(data, (np.ndarray, np.array)):
            data_type = 'array'

        return data_type

    @property
    def data(self):
        """Return the data as is without conversion, e.g. a root-dict, pandasDF etc."""
        return self._data

    def set_data(self, data, index=None, columns=None, column_alias=None):
        """Set the data and also change index and columns.

        Parameters
        ----------
        data : |data_type|
            The new data
        index : |index_type|
            |index_docstring|
        columns : list(str, str, str,...)
            The columns for the data to use
        column_alias : |column_alias_type|
            |column_alias_docstring|
        """

        if column_alias is not None:
            self.column_alias.update(column_alias)
        self._set_data(data=data, index=index, columns=columns)

    def _set_data(self, data, index=None, columns=None):
        """Set the data, length- and columns-attribute.

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

            self._data = self._make_df(index=self._index)  # No cols, it's set above
        # numpy array
        elif self._data_type == 'array':
            self._data = self._make_df(index=self._index)
            warnings.warn(DeprecationWarning, "Not safe, it's better to use pandas DataFrame")
        else:
            raise NotImplementedError("Other dataformats are not yet implemented")

# TODO: implement pickleable data?
#
#    @property
#    def labels(self):
#        return self._label_dic.get(self.columns)

    def get_weights(self, index=None, normalize=True, **kwargs):
        """Return the weights of the specified indeces or, if None, return all.

        Parameters
        ----------
        normalize : boolean or float > 0
            If True, the weights will be normalized to 1 (the mean is 1).
            If a float is provided, the mean of the weights will be equal
            to *normalize*. So *True* and *1* will yield the same results.
        index : |index_type|
            |index_docstring

        Return
        ------
        out: 1-D pandas Series
            Return the weights as pandas Series
        """
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)
        normalize = 1 if normalize is True else normalize
        second_storage = kwargs.get('second_storage')

        normalize_1 = 1
        normalize_2 = 1

        # HACK
        weights_ratio = normalize

        # TODO: implement if targets are different

        if weights_ratio > 0 and second_storage is not None:
            weights_1 = self.get_weights(index=index)
            weights_2 = second_storage.get_weights()

            sum_weight_1 = float(sum(weights_1))
            sum_weight_2 = float(sum(weights_2))

            ratio_1 = weights_ratio * sum_weight_2 / sum_weight_1
            self.logger.info("ratio_1 = " + str(ratio_1))
            if ratio_1 >= 1:
                ratio_2 = 1.0
            else:
                ratio_2 = 1.0 / ratio_1
                ratio_1 = 1.0

            normalize_1 = ratio_1
            normalize_2 = ratio_2
        elif weights_ratio > 0 and second_storage is None:
            normalize_1 = weights_ratio
        else:
            normalize_1 = normalize_2 = None

        weights_out = self._get_weights(index=index, normalize=normalize_1)

        if dev_tool.is_in_primitive(weights_out, (None, 1)):
            weights_out = pd.Series(data=np.ones(length), index=index) * normalize_1

        if second_storage is not None:
            weights_2 = second_storage.get_weights(normalize=normalize_2)
            weights_out = np.concatenate((weights_out, weights_2))

        return weights_out

    def _get_weights(self, index=None,  normalize=True):
        """Return pandas Series of weights or None, 1."""
        # initialize values
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        # TODO: allow other primitive weights
        if dev_tool.is_in_primitive(self._weights, (None, 1)):
            weights_out = self._weights
            if normalize != 1 or normalize is not True:
                weights_out = pd.Series(np.ones(length), index=index)
            else:
                normalize = False
        elif index is None:
            weights_out = self._weights
        else:
            weights_out = self._weights.loc[index]
        weights_out = copy.deepcopy(weights_out)
        if normalize or normalize > 0:
            normalize = 1 if normalize is True else normalize
            weights_out *= normalize / weights_out.mean()

        return weights_out

    def set_weights(self, sample_weights, index=None):
        """Set the weights of the sample.

        Parameters
        ----------
        sample_weights : |sample_weights_type|
            |sample_weights_docstring|
        index : 1-D array or list or None
            The indeces for the weights to be set. Only the index given will be
            set/used as weights.
        """
        index = self._index if index is None else index

        if isinstance(sample_weights, (str, dict)) and self._data_type == 'root':
            assert ((isinstance(sample_weights, list) and (len(sample_weights) == 1)) or
                    isinstance(sample_weights, str)), "Can only be one branche"
            assert isinstance(self._data, dict), "data should be root-dict but is no more..."
            tmp_root = copy.deepcopy(self._data)
            if isinstance(sample_weights, str):
                sample_weights = {'branches': sample_weights}
            tmp_root.update(sample_weights)

            sample_weights = data_tools.to_ndarray(tmp_root)

        self._set_weights(sample_weights=sample_weights, index=index)

    def _set_weights(self, sample_weights, index=None):
        """Set the weights"""
        index = self.index if index is None else index
        length = len(self) if index is None else len(index)

        if dev_tool.is_in_primitive(sample_weights, (None, 1)):
            if index is None or len(self) == len(index):
                self._weights = 1
                return
            else:
                sample_weights = pd.Series(np.ones(len(index)), index=index)
        #    else:
        #        sample_weights = np.ones(length)
        elif isinstance(sample_weights, pd.Series):
            sample_weights = sample_weights[index]
        else:
            sample_weights = pd.Series(sample_weights, index=index, dtype='f8')

        if len(self) == length and index is None:
            self._weights = sample_weights
        else:
            if dev_tool.is_in_primitive(self._weights, (None, 1)):
                self._weights = pd.Series(np.ones(len(self)), index=self._index)
            self._weights.update(sample_weights)

    def set_root_selection(self, selection, exception_if_failure=True):
        """Set the selection in a root-file. Only possible if a root-file is provided."""
        warnings.warn("Method set_root_selection very unsafe currently!")
        meta_config.warning_occured()
        if self._data_type == 'root':
            self.data['selection'] = selection
            self.set_data(self.data, columns=self.columns)
            self.data_name_addition += "INDEX CRASHED!"
        elif exception_if_failure:
            raise RuntimeError("selection could not be applied, no root-dict")
        else:
            self.logger.error("selection not applied, no root-dict")

    def pandasDF(self, columns=None, index=None):
        """Return a pandas DataFrame representation of the data

        Return a pandas DataFrame.

        Parameters
        ---------
        columns : str
            Arguments for the :py:func:`~root_numpy.root2rec` ls
            function.
        index : |index_type|
            |index_docstring|
        """
        # initialize variables
        index = None if index is None else list(index)
        columns = None if columns is None else data_tools.to_list(columns)

        # create data
        data_out = self._make_df(columns=columns, index=index, copy=True)
        # TODO: leave away below?!
#        if not data_out.index.tolist() == range(len(data_out)):  # if not, convert the indices to
#            data_out.reset_index(drop=True, inplace=True)

        return data_out

    def _make_df(self, columns=None, index=None, copy=False):
        """Return a DataFrame from the internal data. Does some dirty, internal work."""
        # initialize data
        # TODO: remove trailing comment?
        data = self._data  # if dev_tool.is_in_primitive(data) else data
        columns = self.columns if columns is None else data_tools.to_list(columns)
        index = self._index if index is None else data_tools.to_list(index)

        if self._data_type == 'root':
            # update root dictionary
            temp_root_dict = dict(data, **{'branches': columns})
            for key, val in temp_root_dict.items():
                if dev_tool.is_in_primitive(val, None):
                    temp_root_dict[key] = self.data.get(key)
            data = data_tools.to_pandas(temp_root_dict, columns=columns)

        elif self._data_type == 'array':
            data = pd.DataFrame(data, index=index, columns=columns, copy=copy)
        elif self._data_type == 'df':
            if columns is not None:
                data = data[columns]
        else:
            raise NotImplementedError("Unknown/not yet implemented data type")

        assert isinstance(data, pd.DataFrame), "data did not convert correctly"
        data = data if index is None else data.ix[index]

        if isinstance(self.column_alias, dict) and len(self.column_alias) > 0:
            data.rename(columns=self.column_alias, inplace=True, copy=False)

        return data

#    def get_labels(self, columns=None, as_list=False):
#        """Return the human readable branch-labels of the data.
#
#        Parameters
#        ----------
#        columns : list with str or str
#            The labels of the columns to return
#        as_list : boolean
#            If true, the labels will be returned as a list instead of a dict.
#
#        Return
#        ------
#        out : list or dict
#            Return a list or dict containing the labels.
#        """
#        if columns is None:
#            columns = self.columns
#        columns = data_tools.to_list(columns)
#        if as_list:
#            labels_out = [self._label_dic.get(col, col) for col in columns]
#        else:
#            labels_out = {key: self._label_dic.get(key) for key in columns}
#        return labels_out

        # TODO: delete?
#    def set_labels(self, data_labels, replace=False):
#        """Set the human readable data-labels (for the columns).
#
#        Sometimes you want to change the labels(names) of columns. This can be
#        done by passing a dictionary containing the column as key and a
#        human-readable name as value.
#
#        Parameters
#        ----------
#        data_labels : dict
#            It has the form: {column: name}
#        replace : boolean
#        """
#        if data_labels is None:
#            return
#        assert isinstance(data_labels, dict), "Not a dictionary"
#        self._set_data_labels(data_labels=data_labels, replace=replace)
#
#    def _set_data_labels(self, data_labels, replace):
#        """Update the data labels"""
#        if replace:
#            self._label_dic = data_labels
#        else:
#            self._label_dic.update(data_labels)

    def get_targets(self, index=None):
        """Return the targets of the data as a pandas Series."""
        # assing defaults
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        # get targets via internal method
        out_targets = self._get_targets(index=index)

        # create targets if targets are "simpel" for output
        if isinstance(out_targets, (int, float)) or out_targets is None:
            if self._target is None:
                self.logger.warning("Target list consists of None!")
            out_targets = dev_tool.make_list_fill_var([], length, self._target)
            out_targets = pd.Series(out_targets, index=index)

        return out_targets

    def _get_targets(self, index=None):
        """Return targets as pandas Series or primitive type."""
        # assign defaults
        index = self._index if index is None else list(index)
        # length = len(self) if index is None else len(index)

        if index is None or dev_tool.is_in_primitive(self._target, (-1, 0, 1, None)):
            out_targets = self._target
        else:
            out_targets = self._target.loc[index]

        return out_targets

    def set_targets(self, targets, index=None):
        """Set the targets of the data. Either an array-like object or {0, 1}."""

        if not dev_tool.is_in_primitive(targets, (-1, 0, 1, None)):
            assert len(self) == len(targets), "Invalid targets"
        self._set_target(target=targets, index=index)

    def _set_target(self, target, index=None):
        """Set the target. Attention with Series, index must be the same as data-index."""

        index = self._index if dev_tool.is_in_primitive(index) else index
        if isinstance(target, (list, np.ndarray, pd.Series)):
            target = pd.Series(target, index=index, copy=True)
            target.sort_index(inplace=True)
        self._target = target

    def make_dataset(self, second_storage=None, index=None, index_2=None, columns=None,
                     weights_ratio=0, shuffle=False, targets_from_data=False):
        """Create data, targets and weights of the instance (and another one).

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
        index : |index_type|
            The index for the **calling** (the *first*) storage instance.
            |index_docstring|
        index_2 : list(int, int, int, ...)
            The index for the (optional) **second storage instance**.
            |index_docstring|
        columns : list(str, str, str, ...)
            The columns to be used of **both** data-storages.
        weights_ratio : float >= 0
            The (relative) normalization. If a second data storage is provided
            it is assumed (will be changed in future ?)
            that the two storages can be seen as the two different targets.
            If zero, nothing happens. If it is bigger than zero, it
            represents the ratio of the sum of the weights from the first
            to the second storage. If set to 1, they both are equally
            weighted.
            If no second storage is provided, it is the normalization of the
            storage called.

            Ratio := sum(weights_1) / sum(weights_2) with a second storage
            Ratio := sum(weights_1) / mean(weights_1)

        shuffle : boolean or int
            If True or int, the dataset will be shuffled before returned. If an
            int is provided, it will be used as a seed to the pseudo-random
            generator.
        targets_from_data
            OUTDATED, dont use it. Use two datastorage, one labeled 0, one 1
         """
        # initialize values

#        normalize_1 = 1
#        normalize_2 = 1
#
#        if weights_ratio > 0 and second_storage is not None:
#            weights_1 = self.get_weights(index=index)
#            weights_2 = second_storage.get_weights(index=index_2)
#
#            sum_weight_1 = float(sum(weights_1))
#            sum_weight_2 = float(sum(weights_2))
#
#            ratio_1 = weights_ratio * sum_weight_2 / sum_weight_1
#            self.logger.info("ratio_1 = " + str(ratio_1))
#            if ratio_1 >= 1:
#                ratio_2 = 1.0
#            else:
#                ratio_2 = 1.0 / ratio_1
#                ratio_1 = 1.0
#
#            normalize_1 = ratio_1
#            normalize_2 = ratio_2
#        elif weights_ratio > 0 and second_storage is None:
#            normalize_1 = weights_ratio
#        else:
#            normalize_1 = None

        if shuffle is not False:
            index = self.index if index is None else index
            if isinstance(shuffle, int) and shuffle is not True:
                rand_seed = shuffle
                rand_seed_2 = shuffle + 74
            else:
                rand_seed = rand_seed_2 = None
            random.shuffle(index, random=rand_seed)
        data = self.pandasDF(columns=columns, index=index)
        if second_storage is None:
            targets = self.get_targets(index=index)
#        weights = self.get_weights(index=index, normalize=normalize_1)

        if second_storage is not None:
            assert isinstance(second_storage, HEPDataStorage), "Wrong type, not an HEPDataStorage"
            if shuffle is not False:
                index_2 = second_storage.index if index_2 is None else index_2
                random.shuffle(index_2, random=rand_seed_2)
            data_2 = second_storage.pandasDF(columns=columns, index=index_2)
            data = pd.concat((data, data_2), ignore_index=True, copy=False)

            targets_1 = self.get_targets()
            targets_2 = second_storage.get_targets()
            targets = np.concatenate((targets_1, targets_2))

            if max(targets_1) != min(targets_1) or max(targets_2) != min(targets_2) and weights_ratio > 0:
                raise ValueError("Very unfortunately is the case of mixed targets in a HEPDataStorage and weights_ratio"+
                                 "not yet implemented. Please make an issue!")

#            weights_2 = second_storage.get_weights(index=index_2, normalize=normalize_2)

        weights = self.get_weights(normalize=weights_ratio, second_storage=second_storage)

        return data, targets, weights

    def copy_storage(self, columns=None, index=None, add_to_name=" cp"):
        """Return a copy of self (with only some of the columns, indices etc).

        Parameters
        ----------
        columns : str or list(str, str, str, ...)
            The columns which will be in the new storage.
        index : |index_type|
            The indices of the rows (and corresponding weights, targets etc.)
            for the new storage.
            |index_docstring|
        add_to_name : str
            An addition to the data_name_addition of the copy.
        """
        index = self._index if index is None else list(index)
        columns = self.columns if columns is None else columns

        new_data = copy.deepcopy(self.data)

        new_targets = copy.deepcopy(self._get_targets(index=index))
        new_weights = copy.deepcopy(self._get_weights(index=index))
        new_index = copy.deepcopy(index)
        new_column_alias = copy.deepcopy(self.column_alias)

        new_storage = HEPDataStorage(new_data, target=new_targets,
                                     sample_weights=new_weights,
                                     index=new_index,
                                     column_alias=new_column_alias,
                                     data_name=self.data_name,
                                     data_name_addition=self.data_name_addition + add_to_name)
        new_storage.columns = columns
        return new_storage

# TODO: add second data_storage
    def get_LabeledDataStorage(self, columns=None, index=None, shuffle=False):
        """Create and return an instance of class "LabeledDataStorage" from the REP repository.

        Parameters
        ----------
        columns : str or list(str, str, str, ...)
            The columns to use for the LabeledDataStorage.
        index : |index_type|
            |index_docstring|
        shuffle : boolean
            Argument is passed to the LabeledDataStorage. If True, the data
            will be shuffled.

        Return
        ------
        out: LabeledDataStorage instance
            Return a Labeled Data Storage instance created with the data
            from inside this instance.
        """
        index = self.index if index is None else list(index)
        columns = self.columns if columns is None else columns
        random_state = meta_config.randint()
        new_lds = LabeledDataStorage(self.pandasDF(columns=columns, index=index),
                                     target=self.get_targets(index=index),
                                     sample_weight=self.get_weights(index=index),
                                     random_state=random_state, shuffle=shuffle)
        return new_lds

    def make_folds(self, n_folds=10, shuffle=True):
        """Create shuffled train-test folds which can be accessed via :py:meth:`~raredecay.tools.data_storage.HEPDataStorage.get_fold()`.

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
        shuffle : boolean or int
            If True or int, shuffle the data before slicing.
        """
        if not n_folds > 1:
            raise ValueError("Number of folds has to be higher then 1")

        self._fold_index = []

        # split indices of shuffled list
        length = len(self)
        temp_indeces = [int(round(length / n_folds)) * i for i in range(n_folds)]
        temp_indeces.append(length)  # add last index. len(index) = n_folds + 1

        # get a copy of index and shuffle it if True
        temp_index = copy.deepcopy(self._make_index())
        if shuffle is not False:
            random.shuffle(temp_index, random=meta_config.randfloat)
        for i in range(n_folds):
            self._fold_index.append(temp_index[temp_indeces[i]:temp_indeces[i + 1]])

    def get_fold(self, fold):
        """Return the specified fold: train and test data as instance of :py:class:`~raredecay.tools.data_storage.HEPDataStorage`.

        Parameters
        ----------
        fold : int
            The number of the fold to return. From 0 to n_folds - 1

        Return
        ------
        out : tuple(HEPDataStorage, HEPDataStorage)
            Return the *train* and the *test* data in a HEPDataStorage
        """
        assert self._fold_index is not None, "Tried to get a fold but data has no folds." + \
                                             " First create them (make_folds())"
        assert isinstance(fold, int) and fold < len(self._fold_index), "Value of fold is invalid"
        train_index = []
        for i, index_slice in enumerate(self._fold_index):
            if i == fold:
                test_index = copy.deepcopy(index_slice)
            else:
                train_index += copy.deepcopy(index_slice)
        n_folds = len(self._fold_index)
        test_DS = self.copy_storage(index=test_index)
        test_DS._fold_status = (fold, n_folds)
        # + 1 human-readable
        test_DS.fold_name = "test set fold " + str(fold + 1) + " of " + str(n_folds)
        train_DS = self.copy_storage(index=train_index)
        train_DS._fold_status = (fold, n_folds)
        train_DS.fold_name = "train set fold " + str(fold + 1) + " of " + str(n_folds)
        return train_DS, test_DS

    def get_n_folds(self):
        """Return how many folds are currently availabe or 0 if no folds have been created.

        Return
        ------
        out : int
            The number of folds which are currently available.
        """
        return 0 if self._fold_index is None else len(self._fold_index)

    def plot_correlation(self, second_storage=None, figure=None, columns=None,
                         method='pearson', plot_importance=5):
        """
        .. warning:: does not support weights. Maybe in the future.

        Plot the feature correlation for the data (combined with other data)

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
        from statsmodels.stats.weightstats import DescrStatsW
        columns = self.columns if columns is None else columns

        data_name = self.name
        if second_storage is not None:
            data_name += " and " + second_storage.name
        data, _tmp, weights = self.make_dataset(second_storage=second_storage,
                                                shuffle=True, columns=columns)
        del _tmp
        out.save_fig(figure, importance=plot_importance)
        ds = DescrStatsW(data.as_matrix(), weights=weights)
        correlation = ds.cov
        correlation = data.corr(method=method)
        corr_plot = sns.heatmap(correlation.T)

        corr_plot.set_title("Correlation of " + data_name)

        # turn the axis label
        for item in corr_plot.get_yticklabels():
            item.set_rotation(0)

        for item in corr_plot.get_xticklabels():
            item.set_rotation(90)

        return correlation

    def plot(self, figure=None, columns=None, index=None, title=None, sub_title=None,
             data_name=None, bins=None, log_y_axes=False, plot_range=None, x_label=None,
             y_label="probability density", sample_weights=None, importance=3,
             see_all=False, hist_settings=None, figure_kwargs=None):
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
        columns : str or list(str, str, str, ...)
            The columns of the data to be plotted. If None, all are plotted.
        index : |index_type|
            |index_docstring|
        title : str
            | The title of the whole plot (NOT of the subplots). If several
              titles for the same figures are given, they will be *concatenated*.
            | So for a "simple" title, specify the title only once.
        data_name:
            | Additional, (to the *data_name* and *data_name_addition*), human-
              readable name for the legend.
            | Examples: "before cut", "after cut" etc
        bins : int
            Number of bins to plot.
        log_y_axes : boolean
            If True, the y-axes will be scaled logarithmically.
        plot_range : tuple (float, float) or None
            The lower and upper range of the bins. If None, 99.98% of the data
            will be plottet automatically.
        sample_weights : pandas Series
            The weights for the data, how "high" a bin is. Actually, how much
            it should account for the whole distribution or how "often" it
            occures. If None is specified, the weights are taken from the data.
        importance : |importance_type|
            |importance_docstring|
        see_all : boolean
            If True, all data (not just 99.98%) will be plotted.
        hist_settings : dict
            A dictionary containing the settings as keywords for the
            :py:func:`~matplotlib.pyplot.hist()` function.

        """
# ==============================================================================
#        initialize values
# ==============================================================================
        if sample_weights is None:
            sample_weights = self._get_weights(index=index)
            if dev_tool.is_in_primitive(sample_weights, 1):
                sample_weights = None
        figure_kwargs = {} if figure_kwargs is None else figure_kwargs

        # update hist_settings
        if dev_tool.is_in_primitive(hist_settings, None):
            hist_settings = {}
        if isinstance(hist_settings, dict):
            hist_settings = dict(meta_config.DEFAULT_HIST_SETTINGS, **hist_settings)
        if bins is not None:
            hist_settings['bins'] = bins
        if plot_range is not None:
            hist_settings['range'] = plot_range

        # create data
        data_plot = self.pandasDF(columns=columns, index=index)
        columns = data_plot.columns.values
        self.logger.debug("plot columns from pandasDataFrame: " + str(columns))

        # set the right number of rows and columns for the subplot
        subplot_col = int(math.ceil(math.sqrt(len(columns))))
        subplot_row = int(math.ceil(float(len(columns)) / subplot_col))

        # assign a free figure if argument is None
        if dev_tool.is_in_primitive(figure, None):
            while True:
                safety = 0
                figure = self.__figure_number + 1
                self.__figure_number += 1
                assert safety < meta_config.MAX_FIGURES, "stuck in an endless while loop"
                if figure not in self.__figure_dic.keys():
                    x_limits_col = {}
                    # TODO: improve figure dict with title....
                    self.__figure_dic.update({figure: x_limits_col, str(figure) + '_title': ""})
                    break
        elif figure not in self.__figure_dic.keys():
            x_limits_col = {}
            self.__figure_dic.update({figure: x_limits_col, str(figure) + '_title': ""})
        out_figure = out.save_fig(figure, importance=importance,
                                  figure_kwargs=figure_kwargs, **cfg.save_fig_cfg)

        # create a label
        label_name = data_tools.obj_to_string([self._name[0], self._name[1],
                                               data_name], separator=" - ")
        self.__figure_dic[str(figure) + '_title'] += "" if title is None else title
        plt.suptitle(self.__figure_dic.get(str(figure) + '_title'), fontsize=self.supertitle_fontsize)

# ==============================================================================
#       Start plotting
# ==============================================================================
        # plot the distribution column by column
        for col_id, column in enumerate(columns, 1):
            # create sub title
            sub_title_tmp = column if sub_title is None else sub_title
            x_label = "" if x_label is None else x_label

            # only plot in range x_limits, otherwise the plot is too big
            x_limits = self.__figure_dic.get(figure).get(column)
            lower, upper = np.percentile(np.hstack(data_plot[column]),
                                         [0.01, 99.99])
            if dev_tool.is_in_primitive(x_limits, None):
                x_limits = (lower, upper)
            elif see_all:  # choose the maximum range. Bins not nicely overlapping.
                x_limits = (min(x_limits[0], lower), max(x_limits[1], upper))
            if 'range' in hist_settings:
                x_limits = hist_settings.pop('range')
            self.__figure_dic[figure].update({column: x_limits})

            plt.subplot(subplot_row, subplot_col, col_id)
            plt.hist(data_plot[column], weights=sample_weights, log=log_y_axes,
                     range=x_limits, label=label_name, **hist_settings)

            # set labels, titles...
            plt.title(sub_title_tmp)
            ha = 'center'
            plt.xlabel(x_label, ha=ha, position=(0.5, 0))
            if y_label is not None:
                plt.ylabel(y_label, ha=ha, position=(0, 0.5))

        plt.legend()
        return out_figure

    def plot_parallel_coordinates(self, columns=None, figure=0, second_storage=None):
        """Plot the parallel coordinates.

        .. warning::
            No weights supported so far!
        """

        data, targets, weights = self.make_dataset(second_storage=second_storage,
                                                   columns=columns)
        targets.name = 'targets'
        data = pd.concat([data, targets], axis=1)
        out_figure = out.save_fig(figure)
        pd.tools.plotting.parallel_coordinates(data, 'targets')

        return out_figure

    def plot2Dhist(self, x_columns, y_columns=None):
        """Plot a 2D hist of x_columns vs itself or y_columns.

        .. warning:: this can produce A LOT of plots! (x_columns * y_columns)

        Parameters
        ----------
        x_columns : list(str, str, str,...)
            The x columns to plot agains
        y_columns : list(str, str, str,...)
            The y columns to plot agains
        """

        x_columns = self.columns if x_columns == 'all' else x_columns
        y_columns = self.columns if y_columns == 'all' else y_columns
        y_columns = x_columns if y_columns is None else y_columns

        for x_col in x_columns:
            for y_col in y_columns:
                df = self.pandasDF(columns=[x_col, y_col])
                df.plot.hexbin(x_col, y_col, gridsize=30)


    def plot2Dscatter(self, x_branch, y_branch, dot_scale=20, color='b', figure=None):
        """Plot two columns against each other to see the distribution.

        The dots size is proportional to the weights, so you have a good
        overview on the data and the weights.

        Parameters
        ----------
        x_branch : str
            The x column to plot
        x_branch : str
            Thy y column to plot
        dot_scale : int or float
            The overall scaling factor for the dots
        color : str
            A valid (matplotlib.pyplot-compatible) color
        figure : str or int or figure
            The figure to be plotted in

        Return
        ------
        out : figure
            Return the figure
        """
        # TODO: make nice again
        out_figure = out.save_fig(figure)
        weights = self.get_weights()
        assert len(weights) == len(self), "Wrong length of weigths"
        size = weights * dot_scale
        temp_label = data_tools.obj_to_string([i for i in self._name])
        plt.scatter(self.pandasDF(columns=x_branch),
                    self.pandasDF(columns=y_branch), s=size, c=color,
                    alpha=0.5, label=temp_label)
        plt.xlabel(x_branch)
        plt.ylabel(y_branch)
        plt.legend()

        return out_figure

# TODO: add correlation matrix

#if __name__ == '__main__':
#
#    n_tested = 0
#
#    b = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43], [51, 52, 53]])
#    a = pd.DataFrame(b, columns=['one', 'two', 'three'], index=[1, 2, 11, 22, 33])
#    c = copy.deepcopy(a)
#    storage = HEPDataStorage(a, index=[1, 2, 11, 22], target=[1, 1, 1, 0],
#                             sample_weights=[1, 2, 1, 0.5],
#                             data_name="my_data", data_name_addition="and addition")
#    n_tested += 1
#
#    d = a.loc[[1, 2, 11, 22]]
#    pd1 = storage.pandasDF()
#
#    t1 = all(pd1.reset_index(drop=True) == d.reset_index(drop=True))
#    t2 = True
#    t3 = all(storage.get_targets(index=[1, 11, 22]) == np.array([1, 1, 0]))
#    works1 = t1 and t2 and t3
#
#    print "storage with DataFrame works:", works1
#
#    DATA_PATH = '/home/mayou/Big_data/Uni/decay-data/'
#    all_branches = ['B_PT', 'nTracks', 'nSPDHits',
#                    'B_FDCHI2_OWNPV', 'B_DIRA_OWNPV',
#                    'B_IPCHI2_OWNPV', 'l1_PT', 'l1_IPCHI2_OWNPV', 'B_ENDVERTEX_CHI2',
#                    'h1_IPCHI2_OWNPV', 'h1_PT', 'h1_TRACK_TCHI2NDOF'
#                    ]
#    mc_file = 'CUT-Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root'
#    cut_Bu2K1Jpsi_mc = dict(
#        filenames=DATA_PATH+'cut_data/'+mc_file,
#        treename='DecayTree',
#        branches=all_branches
#
#    )
#
#    root_data = dict(data=cut_Bu2K1Jpsi_mc,
#                     sample_weights=None,
#                     data_name="B->K1 J/Psi monte-carlo",
#                     data_name_addition="cut"
#                     )
#
#    storage2 = HEPDataStorage(**root_data)
#
#    storage2.plot_correlation(storage2)
#
#    storage3 = storage2.copy_storage(index=[3, 5, 7, 9], columns=['B_PT', 'nTracks'])
#    df11 = storage3.pandasDF()
#    storage3.set_weights(sample_weights=[1, 4, 1, 0.5])
#
#    storage3.make_folds(4)
#    train, test = storage3.get_fold(1)
#    print train.pandasDF(), "\n", test.pandasDF()
#    train.make_folds(3)
#    train1, test1 = train.get_fold(1)
#    print train1.pandasDF(), "\n", test1.pandasDF()
#    t21 = isinstance(storage2.pandasDF(), pd.DataFrame)
#    print "t21 = ", t21
#
#    t22 = True
#    works2 = t21 and t22
#    works = works1 and works2
#    print "DataFrame works:", works
#    plt.show()
#
#    print "Selftest finished, tested " + str(n_tested) + " functions."

#==============================================================================
# Docs
#==============================================================================
data_storage_docstring = """
.. |data_type| replace:: root-tree dict (:py:func:`~raredecay.tools.data_tools.make_root_dict`) or :py:class:`~pd.DataFrame`
.. |sample_weights_type| replace:: :py:class:`~pd.Series` or :py:class:`~np.array`
    or int {1} or str/dict for root-trees (:py:func:`~raredecay.tools.data_tools.make_root_dict`)
.. |sample_weights_docstring| replace::
            The new weights for the dataset.
            If the new weights are a pandas Series, the index must match the
            internal index
            If the data is a root-tree file,
            a string (naming the branche) or a whole root-dict can be given,
            pointing to the weights stored.
.. |index_type| replace:: list or :py:class:`~np.array`
.. |index_docstring| replace:: The index of the data to use.
.. |column_alias_type| replace:: dict{str: str, str: str, ...}
.. |column_alias_docstring| replace::
            To change the name of a branch. The argument should be a dict looking like
            {'current_branch_name_in_root_tree/DataFrame': 'desired_name'}.
            The current_branch has to exist in the root-tree or DataFrame,
            the desired_name can be anything.
"""
import sys
sys.modules[__name__].__doc__ += data_storage_docstring
