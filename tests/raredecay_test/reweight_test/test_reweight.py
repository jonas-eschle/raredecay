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
from raredecay.analysis.physical_analysis import reweight as reweight_old
from raredecay.analysis.reweight import reweight as reweight_new
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
        self.true_gb_weights = pd.Series([-24.9192258068, -2.25116226308, -3.22076111404,
                                          -4.93817412146, 25.9833308541, 2.83962161081,
                                          16.6763580572, 30.8389749894, -3.35908150636,
                                          0.627851241286, 3.43736060159, 32.7060528641,
                                          -19.5328085713, -1.96850487673, -7.71123786328,
                                          -24.4674375816, -41.4042338273, 0.792619175766,
                                          -27.9289421951, -0.207603845686, 39.0766262061,
                                          2.57770310211, 25.805549292, 3.61442185458, -28.2901650079,
                                          3.77905799017, -1.43777525354, 35.1814312012,
                                          6.11600053672, 8.00626007307, -9.04379227715,
                                          -2.89161219918, -16.2540488396, 11.3587687506,
                                          -0.970106705397, -4.00144854352, 3.87037047772,
                                          7.10770031986, 14.1677091124, -44.4580421274,
                                          6.88058755457, -11.8505732899, 3.32587211618,
                                          30.8317720704, -2.21139923505, 27.5490762246,
                                          -12.5427179617, 7.01536573235, -1.74430982892,
                                          -2.56127716681]

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
        self.true_gb_weights = pd.Series([-0.790915315041, 11.6273013962, 3.05537198226,
                                          2.3812568727, -12.4748803393, 9.87659819878, 1.52595651876,
                                          -9.09567746335, 3.21689540896, -13.6776126785,
                                          9.51508482728, -9.02433029077, 10.6885831514,
                                          -3.30291328143, 2.2761108472, -7.47044588391,
                                          8.16649988774, -1.92729028418, 3.10207408949,
                                          0.0998992541448, -11.1849674748, 2.04810859091,
                                          -4.12157871021, 8.56124409625, -1.9317216852,
                                          -0.509058882833, 13.7833807774, 9.6762237718,
                                          3.19353499461, -1.90416589073, -1.52219131824,
                                          -3.32615860027, -5.12077986943, -2.24089843434,
                                          13.025918477, -1.30787299274, -4.98126467145,
                                          0.276302687086, 0.723708380621, 0.144781389219,
                                          5.30314213455, 0.565163873132, 5.0978065514,
                                          -5.17483699138, 14.5843995131, 1.67431649351,
                                          1.62602808468, -3.18735109135, 3.12891466973,
                                          5.33230522957]

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
        self.true_gb_weights = pd.Series([1.0587115324, -5.01316246443, 5.38004589472, 5.187239738,
                                          2.60007460796, 1.76754658897, 8.60745708123,
                                          -0.00153301257559, -5.77617499592, 2.19671445612,
                                          10.1951405388, -4.95943543, -1.76657746373, 2.03529727971,
                                          -0.263449240735, -4.80081360387, 5.48285740313,
                                          11.3387381277, 0.958553542345, -4.41845416315,
                                          -7.57023684844, 1.2608063585, 5.39498121829, 2.32228796379,
                                          -0.0117482949419, -2.87994813294, -5.3377327848,
                                          4.07968127504, 4.46587726891, -7.55444725652,
                                          7.77002216716, 2.80565904667, 6.84024078354,
                                          -6.55938274253, -6.67519422386, -0.728057907124,
                                          2.45725900289, 8.77935665556, 3.45082042106,
                                          -1.18605652916, -5.01617560313, 6.01138344625,
                                          -4.71480611919, 7.95881654857, -1.38987347887,
                                          7.7736538936, 3.31814180312, -0.669538239015,
                                          -2.5384465836, -1.66611952549]

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
        self.true_gb_weights = pd.Series([-2.01740913875, 1.44651111353, -9.90270215364,
                                          7.59211542236, 2.19672581333, 3.76270584851, 2.81004822096,
                                          2.57993730371, 5.32207101898, -3.02010842452,
                                          2.19514069382, -0.656314333333, 1.74467543269,
                                          2.77205765579, -6.70977838796, -3.70313179528,
                                          7.49588370022, 7.92094077344, -4.18331704508,
                                          2.08583648965, 6.61670417561, -4.1958608589,
                                          -0.678585771892, -1.00939256532, -3.24815915566,
                                          5.25609374062, 6.54616090126, -6.34335027989,
                                          5.76642361098, 4.90405225323, 4.10237397243,
                                          -8.34548196601, -1.97962908026, -3.87355976361,
                                          -0.78709388321, 2.41498531738, -2.53296490055,
                                          6.24569627716, 4.39466631946, -2.94095681546,
                                          5.44579016364, 0.758662027231, 0.575822509651,
                                          2.12117976949, 7.82858047411, 2.5234861443, 3.31006620595,
                                          -0.647754084526, -0.443327294055, -1.5165156516]

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
