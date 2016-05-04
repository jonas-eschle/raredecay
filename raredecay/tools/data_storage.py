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

from root_numpy import root2rec
from rep.data.storage import LabeledDataStorage

from raredecay.tools import data_tools, dev_tool
from raredecay.globals_ import out

import importlib
from raredecay import meta_config
cfg = importlib.import_module(meta_config.run_config)
modul_logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


class HEPDataStorage(object):
    """ A wrapper around pandas.DataFrame and an extension to the
    LabeledDataStorage.



    """
    __figure_number = 0
    __figure_dic = {}

    def __init__(self, data, index=None, target=None, sample_weights=None,
                 data_name=None, data_name_addition=None, data_labels=None,
                 add_label=False, hist_settings=None, supertitle_fontsize=18,
                 logger=None):
        """Initialize instance and load data

        Parameters
        ----------
        data : root-tree dict
            Dictionary which specifies all the information to convert a root-
            tree to an array. Directly given to :py:func:`~root_numpy.root2rec`
        .. note:: Will be also arrays or pandas DataFrame in the future
        index : 1-D array-like
            The indices of the data that will be used.
        target : list or 1-D array or int {0, 1}
            Labels the data for the machine learning. Usually the y.
        sample_weights : 1-D array or 1
            Contains the weights of the samples.
        .. note:: If None specified, 1 will be assumed for all.
        data_name : str
            | Name of the data, human-readable. Displayed in the title of \
            plots.
            | *Example: 'Bu2K1piee_mc', 'beta-decay_realData' etc.*
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
        add_label : boolean
            If true, the human-readable labels will be added to the branch name
            shows in the plot instead of replaced.
        hist_settings : dict
            Dictionary with the settings for the histogram plot function
            :func:`~matplotlip.pyplot.hist`
        supertitle_fontsize : int
            The size of the title of several subplots (data_name, _addition)
        """
        # initialize logger
        self.logger = modul_logger if logger is None else logger

        # initialize data
        # TODO: not implemented: needed?
        # self._data_pandas = None
        self._root_dict = data
        self._index = index if index is None else list(index)
        self._fold_index = None  # list with indeces of folds
        self._fold_status = None  # tuple (my_fold_number, total_n_folds)

        # data name
        self._fold_name = None
        self._name = [data_name, data_name_addition, self._fold_name]

        # initialize targets
        if isinstance(target, (list, np.ndarray, pd.Series)):
            target = pd.Series(target, index=index, copy=True)
        self._target_label = target

        # data-labels human readable
        data_labels = {} if data_labels is None else data_labels
        self.add_label = add_label
        self._label_dic = {col: col for col in self._root_dict.get('branches')}
        self._label_dic.update(data_labels)

        # define length of object
        if dev_tool.is_in_primitive(index):
            temp_root_dict = copy.deepcopy(self._root_dict)
            temp_branch = temp_root_dict.pop('branches')  # remove to only use one branch
            temp_branch = dev_tool.make_list_fill_var(temp_branch)
            self._length = len(root2rec(branches=temp_branch[0], **temp_root_dict))
        else:
            self._length = len(index)

        # initialize weights
        if not dev_tool.is_in_primitive(sample_weights, (None, 1)):
            #assert len(sample_weights) == self._length
            sample_weights = pd.Series(sample_weights, index=index, copy=True)
        self.weights = sample_weights

        # plot settings
        if dev_tool.is_in_primitive(hist_settings, None):
            hist_settings = meta_config.DEFAULT_HIST_SETTINGS
        self.hist_settings = hist_settings
        self.supertitle_fontsize = supertitle_fontsize

    def __len__(self):
        return self._length

    def get_rootdict(self, return_index=False):
        """Return the root-dictionary if available, else None"""
        if return_index:
            return self._root_dict, self._index
        else:
            return self._root_dict

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
            Return the train and the test data in a HEPDataStorage
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
        test_DS._fold_name = "test set of fold " + str(fold) + " of " + str(n_folds)
        train_DS = self.copy_storage(index=train_index)
        train_DS._fold_status = (fold, n_folds)
        train_DS._fold_name = "train set of fold " + str(fold) + " of " + str(n_folds)
        return train_DS, test_DS


    def get_n_folds(self):
        """Return how many folds are currently availabe or None if no folds
        created

        Return
        ------
        out : int
            The number of folds which are currently available.
        """
        return None if self._fold_index is None else len(self._fold_index)


    def get_weights(self, normalize=True, index=None, **kwargs):
        """Return the weights of the specified indeces or, if None, return all.

        Parameters
        ----------
        normalize : boolean
            If True, the weights will be normalized to 1 (the average is 1).
        index : 1-D list
            List of index to get the weights from.
        Return
        ------
        out: 1-D numpy array
            Return the weights in an array
        """
        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        if dev_tool.is_in_primitive(self.weights, (None, 1)):
            normalize = False
            if kwargs.get('inter', False):
                weights_out = self.weights
            else:
                weights_out = dev_tool.fill_list_var([], length, 1)
                weights_out = data_tools.to_ndarray(weights_out)
        elif index is None:
            weights_out = data_tools.to_ndarray(self.weights)
        else:
            weights_out = data_tools.to_ndarray(self.weights[index])


        if normalize:
            eps = 0.00001
            counter = 0
            while not (1-eps < weights_out.mean() and weights_out.mean() < 1 + eps):
                weights_out *= weights_out.size/weights_out.sum()
                counter += 1
                if counter > 100:  # to prevent endless loops
                    self.logger.warning("Could not normalize weights. Mean(weights): " + str(np.mean(weights_out)))
                    break
        return weights_out

    def set_weights(self, sample_weights):
        """Set the weights of the sample.

        Parameters
        ----------
        sample_weights : 1-D array or list or int {1}
            The new weights for the dataset.
        """
        assert (len(sample_weights) == len(self) or dev_tool.is_in_primitive(
                sample_weights, (None, 1)), "Invalid weights")
        if not dev_tool.is_in_primitive(sample_weights, (None, 1)):
            sample_weights = pd.Series(sample_weights, index=self._index)
        self.weights = sample_weights

    def extend(self, branches, treename=None, filenames=None, selection=None):
        """Add the branches as columns to the data

        """
        warnings.warn("Function 'extend' not yet implemented")

    def pandasDF(self, branches=None, treename=None, filenames=None,
                 selection=None, index=None):
        """Convert the data to pandas or cut an already existing data frame and
        return it.

        Return a pandas DataFrame

        Parameters
        ---------
        branches, treename, filenames, selection : str
            Arguments for the :py:func:`~root_numpy.root2rec` function.
        index : 1-D list
            The index from the root-branche to be used. If None, all indices
            will be used (all the HEPDataStorage instance was created with).
        """
        index = self._index if index is None else list(index)

        if isinstance(branches, str):
            branches = [branches]
        temp_root_dict = {'branches': branches, 'treename': treename,
                          'filenames': filenames, 'selection': selection}
        for key, val in temp_root_dict.iteritems():
            if dev_tool.is_in_primitive(val, None):
                temp_root_dict[key] = self._root_dict.get(key)

        if index is None:  # if None, return all indices
            data_out = data_tools.to_pandas(temp_root_dict)
        else:
            # TODO: horrible all the conversion? How to change DF indexes?
            data_out = data_tools.to_pandas(temp_root_dict)
            data_out = np.array(data_out.iloc[index])
            # apply "normal" indices for the output array
            data_out = pd.DataFrame(data_out, index=range(len(data_out)))

        #if len(temp_root_dict['branches']) == 1:  # pandas has naming "problems" if only 1 branch
        data_out.columns = temp_root_dict['branches']
        return data_out

    def get_labels(self, branches=None, as_list=False):
        """Return the human readable branch-labels of the data.

        Parameters
        ----------
        branches : list with str or str
            The labels of the branches to return
        as_list : boolean
            If true, the labels will be returned as a list instead of a dict.
        Return
        ------
        out : list or dict
            Return a list or dict containing the labels.
        """
        if branches is None:
            branches = self._root_dict.get('branches')
        branches = dev_tool.make_list_fill_var(branches)
        if as_list:
            labels_out = [self._label_dic.get(col, col) for col in branches]
        else:
            labels_out = {key: self._label_dic.get(key) for key in branches}
        return dev_tool.make_list_fill_var(labels_out)

    def get_name(self, add_str=None, separator=None):
        """Return the human-readable name of the data as a string

        Parameters
        ----------
        add_str : obj with string representation
            To be added after the data name.
        separator : str
            Separates the different name, subname, add_str etc.
        """

        out_str = data_tools.obj_to_string(self._name, separator=separator)
        out_str = data_tools.obj_to_string([out_str, add_str], separator=separator)

        return out_str

    def get_targets(self, index=None, **kwargs):
        """Return the targets of the data **as a numpy array**."""

        index = self._index if index is None else list(index)
        length = len(self) if index is None else len(index)

        if dev_tool.is_in_primitive(self._target_label, (0, 1, None)):
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

        return np.array(out_targets)

    def copy_storage(self, branches=None, index=None):
        """Return a copy of self with only some of the columns (and therefore \
        labels etc) or indices.

        Parameters
        ----------
        branches : str or list(str, str, str, ...)
            The branches which will be in the new storage.
        index : 1-D array-like
            The indices of the rows (and corresponding weights, targets etc.)
            for the new storage.
        """
        index = self._index if index is None else list(index)
        new_labels = {}
        new_root_dic = copy.deepcopy(self._root_dict)

        if branches is not None:
            branches = data_tools.to_list(branches)
            new_root_dic['branches'] = branches

        new_targets = self.get_targets(index=index, inter=True)
        new_weights = self.get_weights(index=index, inter=True)

        for column in new_root_dic['branches']:
            new_labels[column] = self._label_dic.get(column)

        new_storage = HEPDataStorage(new_root_dic, target=new_targets,
                                     sample_weights=new_weights,
                                     data_labels=new_labels, index=index,
                                     add_label=self.add_label, data_name=self._name[0],
                                     data_name_addition=self._name[1] + " cp")

        return new_storage

    def get_LabeledDataStorage(self, index=None, random_state=None, shuffle=False):
        """Create and return an instance of class "LabeledDataStorage" from
        the REP repository

        Return
        ------
        out: LabeledDataStorage instance
            Return a Labeled Data Storage instance created with the data
        """
        index = self._index if index is None else list(index)
        new_lds = LabeledDataStorage(self.pandasDF(index=index), target=self.get_targets(index=index),
                                     sample_weight=self.get_weights(index=index),
                                     random_state=random_state, shuffle=shuffle)
        return new_lds

    def plot(self, figure=None, plots_name=None, std_save=True, log_y_axes=False,
             branches=None, index=None, sample_weights=None, data_labels=None,
             hist_settings=None):
        """Draw histograms of the data.


        Parameters
        ----------
        figure : str or int
            The name of the figure. If the figure already exists, the plots
            will be plotted in the same window (can be intentional, for
            example to compare data)
        log_y_axes : boolean
            If True, the y-axes will be scaled logarithmically.
        plots_name:
            Additional, (to the *data_name* and *data_name_addition*), human-
            readable name for the plot.
        data_labels : dict
            Contain the column as key and the value as label

        """
        # update labels
        if dev_tool.is_in_primitive(data_labels, None):
            data_labels = {}
        data_labels = dict(self._label_dic, **data_labels)
        # update weights
        if dev_tool.is_in_primitive(sample_weights, None):
            sample_weights = self.get_weights(index=index)
        assert len(sample_weights) == len(self.get_weights(index=index)), str(
                "sample_weights is not the right lengths")
        # update hist_settings
        if dev_tool.is_in_primitive(hist_settings, None):
            hist_settings = {}
        if isinstance(hist_settings, dict):
            hist_settings = dict(meta_config.DEFAULT_HIST_SETTINGS, **hist_settings)
        data_plot = self.pandasDF(branches=branches, index=index)
        columns = data_plot.columns.values
        self.logger.debug("plot columns from pandasDataFrame: " + str(columns))
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
            self.__figure_dic.update({figure: x_limits_col})
        out_figure = plt.figure(figure, figsize=(20, 30))

        temp_name = data_tools.obj_to_string([i for i in self._name] + [plots_name],
                                             separator=" - ")
        label_name = data_tools.obj_to_string([self._name[0], self._name[1]], separator=" - ")
        plt.suptitle(temp_name, fontsize=self.supertitle_fontsize)

        # plot the distribution column by column
        for col_id, column in enumerate(columns, 1):
            # only plot in range x_limits, otherwise the plot is too big
            x_limits = self.__figure_dic.get(figure).get(column, None)
            if dev_tool.is_in_primitive(x_limits, None):
                x_limits = np.percentile(np.hstack(data_plot[column]),
                                         [0.01, 99.99])
                self.__figure_dic[figure].update({column: x_limits})
            plt.subplot(subplot_row, subplot_col, col_id)
            temp1, temp2, patches = plt.hist(data_plot[column], weights=sample_weights, log=log_y_axes,
                     range=x_limits, label=label_name,#data_labels.get(column),
                     **hist_settings)
            plt.title(column)
        plt.legend()

        if std_save:
            out.save_fig(out_figure, **meta_config.DEFAULT_SAVE_FIGURE)
        return out_figure

    def plot2Dscatter(self, x_branch, y_branch, dot_size=20, color='b', weights=None, figure=0):
        """Plots two branches against each other to see the distribution.

        """
        out_figure = plt.figure(figure)
        if isinstance(weights, (int, long, float)):
            weights = dev_tool.make_list_fill_var(weights, length=len(self),
                                                  var=weights)
        else:
            weights = self.get_weights()
        assert len(weights) == len(self), "Wrong length of weigths"
        size = [dot_size*weight for weight in weights]
        temp_label = data_tools.obj_to_string([i for i in self._name])
        plt.scatter(self.pandasDF(branches=x_branch),
                    self.pandasDF(branches=y_branch), s=size, c=color,
                    alpha=0.5, label=temp_label)
        plt.xlabel(self.get_labels(branches=x_branch, as_list=True))
        plt.ylabel(self.get_labels(branches=y_branch, as_list=True))
        plt.legend()

        return out_figure

# TODO: add correlation matrix










