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
from raredecay.analysis.ml_analysis import reweight_Kfold, reweight_kfold
import raredecay.settings

out = raredecay.settings.initialize(run_name="test reweighting",
                                    no_interactive_plots=True, n_cpu=-2)

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


class TestReweight_kfold(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series([-3.8730237454, -16.9921211423, -17.0423909366,
                                          7.9161673668, -0.409358103889, -10.1803891303,
                                          -0.690369543253, -7.24159221478, 1.84702658396,
                                          -2.77814498734, -5.4401702245, 9.74092043503,
                                          -2.95365316101, 9.73483848163, -7.40902792151,
                                          1.57501385982, 1.37077118036, -18.8422110185,
                                          6.71051564327, 35.1131543333, 1.54546468888, 7.4458369218,
                                          -4.46496203214, 28.5244594021, -1.36187075225,
                                          9.16229875306, 3.18409659397, -7.382125767, 38.6784983633,
                                          -0.441535798508, 5.70328530069, 10.1152992293,
                                          1.74428592554, 2.47944659243, 7.91839240456,
                                          -13.4430459214, -4.49517527727, 17.5357675898,
                                          -8.20942180954, 6.81460297612, 49.2434753713,
                                          15.1099908494, -58.0919274642, 1.27131371141,
                                          6.97894837524, 1.74449834525, -8.60046280412,
                                          0.011014300991, -10.17857737, -28.6978264535]

                                         )

    def test_reweight_kfold(self):
        ds1, ds2 = _create_data(2)
        scores = reweight_kfold(real=ds1, mc=ds2, n_folds=3,
                                columns=reweight_branches, reweighter='gb',
                                reweighter_cfg=reweight_cfg, n_reweights=1,
                                add_weights=True)
        new_weights = scores.pop('weights')
        pdtest.assert_series_equal(self.true_gb_weights, new_weights)


class TestReweightCV(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series([1.20504514743, -7.773011601, 7.17983704168, 2.62101861844,
                                          20.4008323011, -0.842483629763, 4.41463800768,
                                          1.9461656895, 1.02576013286, 2.80971428188, -1.51684460218,
                                          -5.13561242024, -0.78173874975, -0.0478258698123,
                                          2.01908866731, 9.95469336889, -13.8354260415,
                                          4.54181926854, 0.813557842005, -4.37114190811,
                                          -7.73244652185, 3.52538328775, 5.52824687984,
                                          11.9259492004, -2.98607645, 4.54062665766, 15.5742672672,
                                          -2.69626668568, -1.10693945488, -2.56038478336,
                                          -1.99162402215, -5.10373852419, -6.32761236727,
                                          -3.638514158, -5.30229826853, 4.5480708681, 4.09711622965,
                                          -2.3436601653, 2.84308310871, -1.39649594822,
                                          0.565284875553, 8.97770384501, 6.91092653798,
                                          -0.92390285854, -1.35830738296, -2.14712949144,
                                          4.53635191488, 4.59801012528, 5.09031229782,
                                          -10.2740215584]

                                         )

    def test_reweightCV(self):
        ds1, ds2 = _create_data(2)
        scores = reweight_Kfold(real_data=ds1, mc_data=ds2, n_folds=3,
                                columns=reweight_branches, reweighter='gb',
                                meta_cfg=reweight_cfg, n_reweights=1,
                                add_weights_to_data=True)
        new_weights = scores.pop('weights')
        pdtest.assert_series_equal(self.true_gb_weights, new_weights)


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

    def test_reweight(self):
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
