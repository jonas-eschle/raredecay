# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:10:29 2016

@author: mayou
"""
import copy
import math
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from root_numpy import root2rec
try:
    from rep.data.storage import LabeledDataStorage
except ImportError:
    print "Could not import parts of the REP repository. Some functions will\
            be unavailable and raise errors"

from raredecay.tools import data_tools
from raredecay.tools import dev_tool


class HEPDataStorage():
    """ A wrapper around pandas.DataFrame and an extension to the \
    LabeledDataStorage

    """
    __HIST_SETTINGS_DEFAULT = dict(
        bins=40,
        normed=True,
        alpha=0.5  # transparency [0.0, 1.0]
        )
    __figure_number = 0
    __figure_dic = {}

    def __init__(self, data, target=None, sample_weights=None, data_name=None,
                 data_name_addition=None, data_labels=None, add_label=True,
                 hist_settings=None, supertitle_fontsize=18):
        """Load data

        """
        self._name = (data_name, data_name_addition)
        if data_labels is None:
            data_labels = {}
        self.add_label = add_label
        if dev_tool.is_in_primitive(hist_settings, None):
            hist_settings = self.__HIST_SETTINGS_DEFAULT
        self.hist_settings = hist_settings
        self.target_label = target
        self.data_name = data_name
        self._data_pandas = None
        self.root_dict = data
        self.label_dic = {}
        for branch in self.root_dict.get('branches'):
            self.label_dic[branch] = data_labels.get(branch, branch)
        for key, val in data_labels.iteritems():
            self.label_dic[key] = val
        self.weights = sample_weights
        temp_root_dict = copy.deepcopy(self.root_dict)
        temp_branch = temp_root_dict.pop('branches')
        temp_branch = dev_tool.make_list_fill_var(temp_branch)
        self.length = len(root2rec(branches=temp_branch[0],
                                   **temp_root_dict))
        self.logger = dev_tool.make_logger(__name__)
        self.supertitle_fontsize = supertitle_fontsize

    def __len__(self):
        return self.length

    @property
    def data(self):
        raise IOError("Don't access data with my_data = HEPDataStorageInstance\
                      .data. Use one of the built-in methods.")

    @data.setter
    def data(self):
        raise IOError("You cannot set the data atribute manualy. Use a method\
                      or the constructor")

    def get_weights(self, index=None):
        """Return the weights of the specified indeces or, if None, return all.

        """
        if dev_tool.is_in_primitive(self.weights, (None, 1)):
            weights_out = np.array(dev_tool.fill_list_var([], len(self), 1),
                                   copy=False)
        else:
            weights_out = data_tools.to_ndarray(self.weights)
        assert len(weights_out) == len(self), str(
                "data and weights differ in lenght")
        return weights_out

    def set_weights(self, sample_weights):
        """Set the weights of the sample

        """
        assert len(sample_weights) == len(self), "Wrong length of weights"
        self.weights = sample_weights

    def extend(self, branches, treename=None, filenames=None, selection=None):
        """Add the branches as columns to the data

        """
        warnings.warn("Function 'extend' not yet implemented")

    def pandasDF(self, branches=None, treename=None, filenames=None,
                 selection=None, index=None):
        """Convert the data to pandas or cut an already existing data frame and
        return

        """
        if isinstance(branches, str):
            branches = [branches]
        temp_root_dict = {'branches': branches, 'treename': treename,
                          'filenames': filenames, 'selection': selection}
        for key, val in temp_root_dict.iteritems():
            if dev_tool.is_in_primitive(val, None):
                temp_root_dict[key] = self.root_dict.get(key)
        data_out = data_tools.to_pandas(temp_root_dict)
        if len(temp_root_dict['branches']) == 1:
            data_out.columns = temp_root_dict['branches']
        return data_out

    def get_labels(self, branches=None):
        """Return the labels of the data

        """
        if branches is None:
            branches = self.root_dict.get('branches')
        branches = dev_tool.make_list_fill_var(branches)
        labels_out = [self.label_dic.get(col, col) for col in branches]
        return dev_tool.make_list_fill_var(labels_out)

    def get_targets(self):

        if dev_tool.is_in_primitive(self.target_label, (0, 1, None)):
            # complicated? No! target_label could be array -> use any, all etc.
            if self.target_label is None:
                self.logger.warning("Target list consists of None")
            self.target_label = dev_tool.make_list_fill_var([], len(self),
                                                            self.target_label)
        if isinstance(self.target_label, list):
            self.target_label = np.array(self.target_label)
        assert len(self.target_label) == len(self), "Target has wrong lengths"
        return self.target_label

    def remove_data(self):
        """Remove data (columns, indices, labels etc.) from itself. Use only \
        if low on memory, otherwise use copy_data_partial

        """

    def copy_storage(self, branches=None):
        """Return a copy of self with only some of the columns (and therefore \
        labels etc).

        """
        branches = dev_tool.make_list_fill_var(branches)
        new_labels = {}
        if self._data_pandas is not None:
            return None
        else:
            new_root_dic = copy.deepcopy(self.root_dict)
            new_root_dic['branches'] = branches
            for column in branches:
                new_labels[column] = self.label_dic.get(column)
            new_storage = HEPDataStorage(new_root_dic, target=self.get_targets,
                                         sample_weights=self.get_weights(),
                                         data_labels=new_labels,
                                         add_label=self.add_label)
        return new_storage

    def get_LabeledDataStorage(self, random_state=None, shuffle=False):
        """Create and return an instance of class "LabeledDataStorage" from
        the REP repository

        """
        new_lds = LabeledDataStorage(self.pandasDF(),
                                     target=self.get_targets(),
                                     sample_weight=self.get_weights(),
                                     random_state=random_state,
                                     shuffle=shuffle)
        return new_lds

    def plot(self, figure=None, branches=None, index=None, sample_weights=None,
             hist_settings=None, data_labels=None, log_y_axes=False,
             plots_name=None):
        """Draw histograms of the data


        Parameters:
        ----------
        data_labels: dict
            Contain the column as key and the value as label
        """
        if dev_tool.is_in_primitive(data_labels, None):
            data_labels = {}
        data_labels = dict(self.label_dic, **data_labels)
        if dev_tool.is_in_primitive(sample_weights, None):
            sample_weights = self.get_weights()
        assert len(sample_weights) == len(self.get_weights()), str(
                "sample_weights is not the right lengths")
        if dev_tool.is_in_primitive(hist_settings, None):
            hist_settings = {}
        if isinstance(hist_settings, dict):
            hist_settings = dict(self.__HIST_SETTINGS_DEFAULT, **hist_settings)
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
                assert safety < 300, "stuck in an endless while loop"
                if figure not in self.__figure_dic.keys():
                    break
        self.__figure_dic.update({figure: (subplot_col * subplot_row,
                                           len(columns))})
        plt.figure(figure)
        # naming the plot. Ugly!
        temp_name = ""
        temp_first = False
        temp_second = False
        if self._name[0] is not None:
            temp_name = str(self._name[0])
            temp_first = True
        if self._name[1] is not None:
            temp_name += " - " if temp_first else ""
            temp_name += str(self._name[1])
            temp_second = True
        if plots_name is not None:
            temp_name += " - " if temp_first or temp_second else ""
            temp_name += str(plots_name)
        plt.suptitle(temp_name, fontsize=self.supertitle_fontsize)
        # plot the distribution column by column
        for col_id, column in enumerate(columns, 1):
            # only plot in range x_limits, otherwise the plot is too big
            x_limits = np.percentile(np.hstack(data_plot[column]),
                                     [0.01, 99.99])
            plt.subplot(subplot_row, subplot_col, col_id)
            plt.hist(data_plot[column], weights=sample_weights, log=log_y_axes,
                     range=x_limits, label=data_labels.get(column),
                     **hist_settings)
            plt.title(column)
            plt.legend()














