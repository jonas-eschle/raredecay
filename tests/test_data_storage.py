#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:29:38 2016

@author: Jonas Eschle "Mayou36"
"""
from __future__ import division

import os
import random
import string

import numpy as np
import pandas as pd
import math as mt

from raredecay.tools.data_storage import HEPDataStorage
from raredecay.analysis.physical_analysis import add_branch_to_rootfile


n_row = 20
n_col = 5
branches = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
branches = branches[:5]


def create_weights():
    weights1 = np.array(range(1, n_row + 1)) / 10 + 1
    return weights1


def create_data():

    array1 = [[r * 100 + c for c in range(1, n_col + 1)] for r in range(1, n_row + 1)]

    df1 = pd.DataFrame(array1, columns=branches[:n_col])

    return df1


def pandasDF(storage):

    df1 = storage.pandasDF()
    assert len(df1) == n_row


def test_root_storage():

    # create root-file
    while True:
        tmp_str = ''.join(random.choice(string.ascii_letters + string.digits) for _t in range(15))
        filename = 'tmp1' + tmp_str + '.root'
        if not os.path.isfile(filename):
            break
    treename = 'tree1'

    df1 = create_data()

    for name, col in df1.iteritems():
        add_branch_to_rootfile(filename=filename, treename=treename,
                               new_branch=col, branch_name=name)

    root_dict = dict(filenames=filename, treename=treename, branches=branches)
    weights1 = create_weights()
    storage1 = HEPDataStorage(data=root_dict, target=1, sample_weights=weights1)

    # start testing
    pandasDF(storage1)

    # remove root-file at the end
    os.remove(filename)


def test_1():
    assert 1 == 1


#if __name__ == '__main__':
#    test_root_storage()
