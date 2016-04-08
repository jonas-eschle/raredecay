# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:10:29 2016

@author: mayou
"""
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from root_numpy import root2rec

from raredecay.tools import data_tools
from raredecay.tools import dev_tool

class HEPDataStorage():
    """ A wrapper around pandas.DataFrame and an extension to the \
    LabeledDataStorage

    """
    def __init__(self, data, target=None, sample_weights=None, data_name=None,
                 data_labels=None, add_label=True):
        """Load data

        """
        if data_labels is None:
            data_labels = {}
        self.add_label= add_label
        self.target_label=target
        self.data_name = data_name
        self.data_pandas = None
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

    def __len__(self):
        return self.length

    @property
    def data(self):
        print "Don't access data like this! Use one of the built-in methods."

    def get_weights(self, index=None):
        """Return the weights of the specified indeces or, if None, return all.

        """
        if self.weights is None:
            weights_out = np.array(dev_tool.fill_list_var([], len(self), 1),
                                   copy=False)
        else:
            weights_out = data_tools.to_ndarray(self.weights)
        assert len(weights_out) == len(self), str(
                "data and weights differ in lenght")
        return weights_out

    def set_weights(self, weights=None):
        """Set the weights of the sample

        """



    def extend(self, branches, treename=None, filenames=None, selection=None):
        """Add the branches as columns to the data

        """
        pass

    def pandasDF(self, branches=None, treename=None, filenames=None,
                 selection=None):
        """Convert the data to pandas or cut an already existing data frame and
        return

        """
        temp_root_dict = {'branches': branches, 'treename': treename,
                          'filenames': filenames, 'selection': selection}
        for key, val in temp_root_dict.iteritems():
            if val is None:
                temp_root_dict[key] = self.root_dict.get(key)
        data_out = data_tools.to_pandas(temp_root_dict)
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
        if self.target_label in (0, 1):
            self.target_label = dev_tool.make_list_fill_var([], len(self),
                                                            self.target_label)
        if isinstance(self.target_label, list):
            self.target_label = np.array(self.target_label)
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
        if self.data_pandas is not None:
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















