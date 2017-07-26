# Python 2 backwards compatibility overhead START
from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct,
                      open, pow, range, round, str, super, zip)
import sys
import warnings
import raredecay.meta_config

try:
    from future.builtins.disabled import (apply, cmp, coerce, execfile, file, long, raw_input,
                                          reduce, reload, unicode, xrange, StandardError)
    from future.standard_library import install_aliases

    install_aliases()
except ImportError as err:
    if sys.version_info[0] < 3:
        raise err
# Python 2 backwards compatibility overhead END

import copy
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.util.testing as pdtest

import raredecay as rd
import raredecay.settings

rd.settings.set_random_seed(42)
from raredecay.tools.data_storage import HEPDataStorage
from raredecay.analysis.physical_analysis import reweight

all_branches = ['a', 'b', 'c', 'd']
reweight_branches = all_branches[1:]
reweight_cfg = dict(
        n_estimators=5,
        max_depth=3,  # 3-6
        learning_rate=0.1,  # 0.1
        min_samples_leaf=100,  # 100
        loss_regularization=8,  # 3-10
        gb_args=dict(
                subsample=0.8,  # 0.8
                min_samples_split=100  # 200

                )
        )


class TestReweight(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series([-8.33032482327, 9.57672179608, -14.7727690569,
                                          14.7628357496, -6.18016786107, -23.0675978918,
                                          10.4312972042, 5.96365804429, -1.04942441752,
                                          5.30208135724, 16.329972665, -0.722033861109,
                                          11.1870766148, -4.98738236272, 9.68241870658,
                                          -16.3222953217, 8.24840813123, 3.39161654987,
                                          -8.5671162781, 12.9621813728, -2.39537119406,
                                          -13.7654664571, 16.9283396731, -1.9443715525,
                                          -2.73681499232, -8.2333444313, 13.0262769418,
                                          13.9056741553, -5.49654510998, -3.12753335034]

                                         )

    def test_something(self):
        ds1, ds2, ds3 = _create_data()
        scores = reweight(apply_data=ds1, real_data=ds2, mc_data=ds3,
                          columns=reweight_branches, reweighter='gb',
                          reweight_cfg=reweight_cfg, n_reweights=1,
                          apply_weights=True)
        new_weights = scores.pop('weights')
        pdtest.assert_series_equal(self.true_gb_weights, new_weights)


def _create_data(n_storages=3):
    data_storages = []

    for i in range(n_storages):
        data = pd.DataFrame(np.random.normal(0.3 * i, 10 + i, size=[30, 4]), columns=all_branches)
        weights = np.random.normal(size=len(data))
        data_storages.append(HEPDataStorage(data, target=i % 2, sample_weights=weights,
                                            data_name='test storage ' + str(i)))

    return data_storages


if __name__ == '__main__':
    unittest.main()
