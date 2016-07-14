# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:21:16 2016

@author: mayou
"""
from __future__ import division, absolute_import

import math as mt
import numpy as np


def punzi_fom(n_signal, n_background, n_sigma=5):
    """Return the Punzi Figure of Merit metric: S / (sqrt(B) + n_sigma/2)

    The Punzi FoM is mostly used for the detection of rare decays to prevent
    the metric of cutting out all the background and leaving us with only a
    very few signals.

    Parameters
    ----------
    n_signal : int
        Number of signals observed (= tpr; true positiv rate)
    n_background : int
        Number of background observed as signal (= fpr; false positiv rate)
    n_sigma : int or float

    """
    length = 1 if not hasattr(n_signal, "__len__") else len(n_signal)
    if length > 1:
        sqrt = np.sqrt(np.array(n_background))
        term1 = np.full(length, n_sigma/2)
    else:
        sqrt = mt.sqrt(n_background)
        term1 = n_sigma/2
    out = n_signal / (sqrt + term1)
    return out


def precision_measure(n_signal, n_background):
    """Return the precision measure: s / sqrt(s + b)"""
    length = 1 if not hasattr(n_signal, "__len__") else len(n_signal)
    if length > 1:
        sqrt = np.sqrt(np.array(n_signal + n_background))
    else:
        sqrt = mt.sqrt(n_signal + n_background)
    out = n_signal / sqrt
    return out
