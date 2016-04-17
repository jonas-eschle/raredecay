# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:35:20 2016

@author: mayou

This module is not intended to be imported or used somewhere else, but to
reproduce and extract strange behaving code and to try to fix it and/or leave
it as a reminder or unsolved problem.

Everything should happen inside the methods.
"""


def __numpy_none_cmp():
    # WORKS SO FAR
    """When you want to test an object if it is None (or not), you will run
    into problems with arrays, as every element is tested.
    """
    import numpy as np
    if 1 is not None:
        print "True. 1 is not None"
    array = np.array([[1, 2, 3], [4, 5, 6]])
    if array is not None:
        print "array is not None. But you will never see this..."
    else:
        print "this should not be printed"


if __name__ == '__main__':
    __numpy_none_cmp()
