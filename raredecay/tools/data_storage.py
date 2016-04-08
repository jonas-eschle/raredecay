# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:10:29 2016

@author: mayou
"""

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
                 data_labels=None):
        """Load data

        """
        self.data_pandas = None
        self.root_dict = data
        self.auto_label = self.root_dict.get('branches')
        self.weights = sample_weights
        self.temp_root_dict = dict(self.root_dict)
        self.temp_branch = self.temp_root_dict.pop('branches')
        self.temp_branch = dev_tool.make_list_fill_var(self.temp_branch)
        self.length = len(root2rec(branches=self.temp_branch[0],
                                   **self.temp_root_dict))
        del self.temp_branch, self.temp_root_dict

    def __len__(self):
        return self.length

    @property
    def data(self):
        print "Don't access data like this! Use one of the built-in methods."

    def get_weights(self, index=None):
        """Return the weights of the specified indeces or, if None, return all.

        """
        if self.weights is None:
            self.weights = np.array(dev_tool.fill_list_var([], len(self), 1),
                                    copy=False)
        else:
            self.weights = data_tools.to_ndarray
        assert len(self.weights) == len(self), str(
                "data and weights differ in lenght")
        return self.weights

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

    def get_labels(branches=None):
        """Return the labels of the data

        """
        return dev_tool.make_list_fill_var(data_labels)

    def get_targets(self):
        return

    def remove_data(self):
        """Remove data (columns, indices, labels etc.) from itself. Use only \
        if low on memory, otherwise use copy_data_partial

        """

    def copy_data(self, branches=None):
        """Return a copy of self with only the columns (and therefore labels \
        etc)specified

        """
        if self.data_pandas is not None:
            return #to do
        else:
            return # object created with dictionary
