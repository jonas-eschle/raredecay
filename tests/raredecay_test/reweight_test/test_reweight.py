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
from raredecay.analysis.compatibility_reweight import reweight as reweight_old
from raredecay.analysis.reweight import reweight as reweight_new, reweight_kfold
from raredecay.analysis.ml_analysis import reweight_Kfold
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
        self.true_gb_weights = pd.Series([-1.18148889078, 0.579988023094, 3.98407795931,
                                          -7.49513507086, -7.30827012925, -8.51036635642,
                                          -23.6840764933, -2.11357212531, 2.49997877252,
                                          -3.31195533049, -7.09858370363, 1.0050575719,
                                          3.20708294845, -8.43263717156, 11.8894509414,
                                          11.8470351007, 3.54745020637, 6.06204281392, 8.18746119587,
                                          -1.77158826059, 0.4074057162, -6.03636765927,
                                          0.0286548041647, 7.62617545756, -5.00099194276,
                                          -3.20152870044, 19.9625438057, 3.79266871345,
                                          -5.26367394049, 3.17681153217, 1.6975778685, 7.4639939175,
                                          1.08003312512, 5.1026070918, 9.82344832334, 2.22383408588,
                                          0.18341048564, 10.4720296422, -1.81730337063,
                                          11.8803426535, 0.895868744306, -3.95622914652,
                                          -12.1134978916, 4.07965074221, 12.5700743144,
                                          -7.20926216964, 8.35579038587, -5.60906691516,
                                          0.920244359919, 6.56280396574]
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
        self.true_gb_weights = pd.Series([0.225861247101, 3.47106584824, 3.55201851671,
                                          1.83174654539, 1.910758964, 4.73223068427, 0.638806543642,
                                          -3.76215585141, -2.0120071684, 4.31649794692,
                                          7.13997292724, -0.210469139421, 2.52142871769,
                                          0.777620616757, 0.670425598938, 1.31884846681,
                                          -2.21312148442, 2.75284539873, 1.91163651042,
                                          -2.09257090487, 0.991136152856, 13.7389270073,
                                          9.04335896666, 4.96733364937, -5.92756557409,
                                          0.897389926056, -3.44115782605, -1.12157563085,
                                          -18.3761804442, -4.03433675213, -0.464863061397,
                                          -3.47251264946, 9.58801242774, 0.73940360788, 2.7254751585,
                                          5.12502589409, 2.32989568021, 6.45517722509, 6.1312116372,
                                          3.20395147939, -0.028973235088, -8.83298907477,
                                          -0.461769737267, 3.10359804597, -9.0907068588,
                                          -3.21646365093, 6.19076335941, 0.685876680963,
                                          -3.18206883957, 8.2531864516]

                                         )

        def test_reweightCV(self):
            ds1, ds2 = _create_data(2)
            scores = reweight_Kfold(real_data=ds1, mc_data=ds2, n_folds=3,
                                    columns=reweight_branches, reweighter='gb',
                                    meta_cfg=reweight_cfg, n_reweights=1,
                                    add_weights_to_data=True)
            new_weights = scores.pop('weights')
            pdtest.assert_series_equal(self.true_gb_weights, new_weights)


class TestReweightOld(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series([3.0208535119, 2.79644475058, 8.54771569694, -7.40831386824,
                                          4.70495037098, 8.24839138368, -12.3022172603,
                                          2.21098260522, -3.48733448669, -0.563879814047,
                                          2.40482485948, -3.37008517604, -4.67376904524,
                                          0.236227028503, -7.5203132393, -7.51048571481,
                                          4.69912069719, -7.94603197218, 4.60878127293,
                                          3.19967295318, -11.937140188, -4.68661067685,
                                          8.25947141481, -6.78657134511, -3.21221298457,
                                          3.19600426056, 5.4185918824, -1.73985585717,
                                          -4.13288628372, 9.83831396354, 12.0277020093,
                                          6.77061085773, 0.590132493624, -5.59064990731,
                                          -5.74405646588, 3.61414984696, 2.89399625915,
                                          9.69464228649, 4.49906342949, -10.3965756863,
                                          7.27616017116, -6.90343213153, -2.65587033877,
                                          1.73059641457, 13.8037256413, 21.5108292006, 4.19661025267,
                                          -1.26714288636, 3.69709351037, 6.13977630316]

                                         )

    def test_reweight_old(self):
        ds1, ds2, ds3 = _create_data()
        scores = reweight_old(apply_data=ds1, real_data=ds2, mc_data=ds3,
                              columns=reweight_branches, reweighter='gb',
                              reweight_cfg=reweight_cfg, n_reweights=10,
                              apply_weights=True)
        new_weights = scores.pop('weights')
        pdtest.assert_series_equal(self.true_gb_weights, new_weights)


class TestReweightNew(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series([2.36993392272, 3.71455903608, 7.17398143621, 6.980238845,
                                          -9.12549109124, -6.2120231791, 3.41152227951,
                                          3.40324698139, 3.41160454189, 25.5199598051, 3.78149967661,
                                          7.52182952892, 6.31917555567, 4.31472543337,
                                          -2.08829981287, 5.02730700212, -5.11908718767,
                                          -1.56865365255, -3.21498091977, 0.54232296003,
                                          15.331977774, -12.3685060264, 4.54569245655,
                                          -10.6824065756, -3.12601131817, 7.21305793869,
                                          0.425781943944, -7.13883214534, -4.73807270322,
                                          4.50156136717, -4.83784741542, 1.43379171868,
                                          0.301861555992, -4.31611045831, 14.2011887194,
                                          4.1989918084, -13.4142640193, 1.2350475578, -4.38358188765,
                                          5.64640035083, -5.24954761019, -0.759998297705,
                                          3.3449657971, 5.73464250229, -7.95059716373,
                                          -2.21568985903, -3.14597196296, -4.32756235289,
                                          11.6941243787, 2.68254276494]

                                         )

    def test_reweight_new(self):
        ds1, ds2, ds3 = _create_data()
        scores = reweight_new(apply_data=ds1, real=ds2, mc=ds3,
                              columns=reweight_branches, reweighter='gb',
                              reweight_cfg=reweight_cfg, n_reweights=3,
                              add_weights=True)
        new_weights = scores.pop('weights')
        pdtest.assert_series_equal(self.true_gb_weights, new_weights)

    def test_reweight_train_only(self):
        ds1, ds2, ds3 = _create_data()
        scores = reweight_new(apply_data=None, real=ds2, mc=ds3,
                              columns=reweight_branches, reweighter='gb',
                              reweight_cfg=reweight_cfg, n_reweights=5,
                              add_weights=True)
        weights = reweight_new(apply_data=ds1, reweighter=scores['reweighter'],
                               add_weights=True)


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
