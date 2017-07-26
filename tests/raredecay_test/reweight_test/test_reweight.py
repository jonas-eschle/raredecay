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
        loss_regularization=13,  # 3-10
        gb_args=dict(
                subsample=0.8,  # 0.8
                min_samples_split=100  # 200

                )
        )


class TestReweight(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series([11.1925573377, 1.29409941787, -5.29351715165,
                                          1.61949809784, -3.06657651146, 5.21564030033,
                                          6.30442815941, 3.62497738454, 4.95152207042, 5.71218672832,
                                          -2.17168830932, -8.58935569906, 3.22026841575,
                                          8.16877279297, -3.17127927275, 6.93310141522,
                                          5.71327799525, -2.39003629241, 0.798180701026,
                                          3.31795561108, -0.330027080817, 1.97678899848,
                                          -9.12348913801, 3.26305458986, 4.41031975868,
                                          2.19708151147, 1.99297756916, -2.87600493681,
                                          9.61593813442, -2.05702267205, 2.61354183439,
                                          -3.7508110579, 0.230042825237, -2.50446356068,
                                          1.88984393675, -2.11494766847, -8.03334361691,
                                          3.56932810397, 0.517577647019, 2.34765083109,
                                          -7.69155096451, 2.09524298171, 3.53772995268,
                                          2.79215140524, 0.655297610201, 3.83135846626,
                                          1.77902338818, -0.759033207129, 0.557551643512,
                                          -4.0158204761]

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
        data = pd.DataFrame(np.random.normal(0.3 * i, 10 + i, size=[50, 4]), columns=all_branches)
        weights = np.random.normal(size=len(data))
        data_storages.append(HEPDataStorage(data, target=i % 2, sample_weights=weights,
                                            data_name='test storage ' + str(i)))

    return data_storages


if __name__ == '__main__':
    unittest.main()
