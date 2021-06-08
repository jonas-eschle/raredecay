# Python 2 backwards compatibility overhead START
import sys

import pytest

try:
    from future.builtins.disabled import (
        apply,
        cmp,
        coerce,
        execfile,
        file,
        long,
        raw_input,
        reduce,
        reload,
        unicode,
        xrange,
        StandardError,
    )
    from future.standard_library import install_aliases

    install_aliases()
except ImportError as err:
    if sys.version_info[0] < 3:
        raise err
# Python 2 backwards compatibility overhead END

import unittest

import numpy as np
import pandas as pd
import pandas.util.testing as pdtest

# import raredecay as rd

from raredecay.tools.data_storage import HEPDataStorage
from raredecay.analysis.compatibility_reweight import reweight as reweight_old
from raredecay.analysis.reweight import reweight as reweight_new, reweight_kfold
from raredecay.analysis.ml_analysis import reweight_Kfold
import raredecay.settings

raredecay.settings.set_random_seed(42)

out = raredecay.settings.initialize(
    run_name="test reweighting", no_interactive_plots=True, n_cpu=-2
)

all_branches = ["a", "b", "c", "d"]
reweight_branches = all_branches[1:]
reweight_cfg = dict(
    n_estimators=5,
    max_depth=3,  # 3-6
    learning_rate=0.1,  # 0.1
    min_samples_leaf=100,  # 100
    loss_regularization=13,  # 3-10
    gb_args=dict(subsample=0.8, min_samples_split=100),  # 0.8  # 200
)


class TestReweight_kfold(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series(
            [
                -0.8445911463880363,
                0.10392354693854089,
                4.879479245649117,
                1.0640894095667925,
                0.9544838901519904,
                4.355682683575129,
                -4.005817560682537,
                3.1763768734231475,
                -2.127926792187807,
                2.283638075651694,
                0.9822853141337221,
                -0.13729257526778438,
                2.8164731292156384,
                -1.5258383628154988,
                2.1366382819086316,
                -0.613947023233219,
                1.1524348388850445,
                2.878090879070167,
                -0.32933349305753845,
                -2.5309720492380885,
                5.516566780275948,
                3.2140143794584555,
                -0.16303269607011947,
                2.322287555717381,
                2.5113999253510118,
                -1.237371251299992,
                5.545778201177078,
                -1.2877793322848847,
                -2.547687088118431,
                1.3908237012492812,
                1.9959032812505426,
                2.1196318339547453,
                -3.9638059566600994,
                -2.310918705082194,
                -2.613449599328348,
                0.8082753723579998,
                -2.287562413169768,
                3.7296340418359373,
                3.645462008395379,
                0.7456757318529652,
                0.0848644563458703,
                -0.14574648154477018,
                0.313946264807159,
                2.6556493859751833,
                4.480049388431625,
                4.602528251174691,
                2.7697024937462116,
                0.6308628907656573,
                5.377065018674339,
                -2.570644604537956,
            ]
        )

    def test_reweight_kfold(self):
        ds1, ds2 = _create_data(2)
        scores = reweight_kfold(
            real=ds1,
            mc=ds2,
            n_folds=3,
            columns=reweight_branches,
            reweighter="gb",
            reweighter_cfg=reweight_cfg,
            n_reweights=1,
            add_weights=True,
        )
        new_weights = scores.pop("weights")
        pdtest.assert_series_equal(self.true_gb_weights, new_weights)


class TestReweightCV(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series(
            [
                -0.8445911463880363,
                0.10392354693854089,
                4.879479245649117,
                1.0640894095667925,
                0.9544838901519904,
                4.355682683575129,
                -4.005817560682537,
                3.1763768734231475,
                -2.127926792187807,
                2.283638075651694,
                0.9822853141337221,
                -0.13729257526778438,
                2.8164731292156384,
                -1.5258383628154988,
                2.1366382819086316,
                -0.613947023233219,
                1.1524348388850445,
                2.878090879070167,
                -0.32933349305753845,
                -2.5309720492380885,
                5.516566780275948,
                3.2140143794584555,
                -0.16303269607011947,
                2.322287555717381,
                2.5113999253510118,
                -1.237371251299992,
                5.545778201177078,
                -1.2877793322848847,
                -2.547687088118431,
                1.3908237012492812,
                1.9959032812505426,
                2.1196318339547453,
                -3.9638059566600994,
                -2.310918705082194,
                -2.613449599328348,
                0.8082753723579998,
                -2.287562413169768,
                3.7296340418359373,
                3.645462008395379,
                0.7456757318529652,
                0.0848644563458703,
                -0.14574648154477018,
                0.313946264807159,
                2.6556493859751833,
                4.480049388431625,
                4.602528251174691,
                2.7697024937462116,
                0.6308628907656573,
                5.377065018674339,
                -2.570644604537956,
            ]
        )

        def test_reweightCV(self):
            ds1, ds2 = _create_data(2)
            scores = reweight_Kfold(
                real_data=ds1,
                mc_data=ds2,
                n_folds=3,
                columns=reweight_branches,
                reweighter="gb",
                meta_cfg=reweight_cfg,
                n_reweights=1,
                add_weights_to_data=True,
            )
            new_weights = scores.pop("weights")
            pdtest.assert_series_equal(self.true_gb_weights, new_weights)


class TestReweightOld(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series(
            [
                -25.545385335151842,
                24.809757564148676,
                -6.210336870882619,
                -40.42883925946502,
                -37.21340108829993,
                94.58667184700776,
                21.708829790052395,
                -83.31284025316096,
                26.959001638897725,
                -9.528695221312033,
                0.2252060216376917,
                40.130461660011925,
                57.9145490749728,
                -42.035674114297095,
                -1.4557428424736885,
                -26.730093063397767,
                29.21417361329943,
                -11.370436972473696,
                1.4317506344905568,
                -13.229037573765734,
                -5.194153776038895,
                83.43629738382411,
                117.01907076876122,
                18.45724900004109,
                -4.893147189184952,
                -78.10045741630597,
                -37.21962961033722,
                -6.203255407884731,
                -34.999672846733176,
                -29.852363474185665,
                77.10942837893583,
                1.4092027568413132,
                5.786035285020609,
                -32.99910884405319,
                57.636414431430275,
                -10.404929548991348,
                -25.494330224087857,
                -77.37184609875148,
                -2.990710373413925,
                -27.02117644563831,
                82.45388420658254,
                90.25195132405007,
                -60.46236464550644,
                -54.92520811618094,
                -13.52481368281434,
                -39.94200588848087,
                10.64172901202869,
                -15.986999375891335,
                13.731009872959032,
                49.73398129416737,
            ]
        )

    def test_reweight_old(self):
        ds1, ds2, ds3 = _create_data()
        scores = reweight_old(
            apply_data=ds1,
            real_data=ds2,
            mc_data=ds3,
            columns=reweight_branches,
            reweighter="gb",
            reweight_cfg=reweight_cfg,
            n_reweights=10,
            apply_weights=True,
        )
        new_weights = scores.pop("weights")
        pdtest.assert_series_equal(self.true_gb_weights, new_weights)


class TestReweightNew(unittest.TestCase):
    def setUp(self):
        self.true_gb_weights = pd.Series(
            [
                2.36993392272,
                3.71455903608,
                7.17398143621,
                6.980238845,
                -9.12549109124,
                -6.2120231791,
                3.41152227951,
                3.40324698139,
                3.41160454189,
                25.5199598051,
                3.78149967661,
                7.52182952892,
                6.31917555567,
                4.31472543337,
                -2.08829981287,
                5.02730700212,
                -5.11908718767,
                -1.56865365255,
                -3.21498091977,
                0.54232296003,
                15.331977774,
                -12.3685060264,
                4.54569245655,
                -10.6824065756,
                -3.12601131817,
                7.21305793869,
                0.425781943944,
                -7.13883214534,
                -4.73807270322,
                4.50156136717,
                -4.83784741542,
                1.43379171868,
                0.301861555992,
                -4.31611045831,
                14.2011887194,
                4.1989918084,
                -13.4142640193,
                1.2350475578,
                -4.38358188765,
                5.64640035083,
                -5.24954761019,
                -0.759998297705,
                3.3449657971,
                5.73464250229,
                -7.95059716373,
                -2.21568985903,
                -3.14597196296,
                -4.32756235289,
                11.6941243787,
                2.68254276494,
            ]
        )

    def test_reweight_new(self):
        ds1, ds2, ds3 = _create_data()
        scores = reweight_new(
            apply_data=ds1,
            real=ds2,
            mc=ds3,
            columns=reweight_branches,
            reweighter="gb",
            reweight_cfg=reweight_cfg,
            n_reweights=3,
            add_weights=True,
        )
        new_weights = scores.pop("weights")
        pdtest.assert_series_equal(self.true_gb_weights, new_weights)

    def test_reweight_new_unnormalized(self):
        ds1, ds2, ds3 = _create_data()
        scores = reweight_new(
            apply_data=ds1,
            real=ds2,
            mc=ds3,
            columns=reweight_branches,
            reweighter="gb",
            reweight_cfg=reweight_cfg,
            n_reweights=3,
            add_weights=True,
            normalize=False,
        )
        new_weights = scores.pop("weights")

        scores = reweight_new(
            apply_data=ds1,
            real=ds2,
            mc=ds3,
            columns=reweight_branches,
            reweighter="gb",
            reweight_cfg=reweight_cfg,
            n_reweights=3,
            add_weights=True,
            normalize=True,
        )
        new_weights_normed = scores.pop("weights")

        scores = reweight_new(
            apply_data=ds1,
            real=ds2,
            mc=ds3,
            columns=reweight_branches,
            reweighter="gb",
            reweight_cfg=reweight_cfg,
            n_reweights=3,
            add_weights=True,
            normalize=10,
        )
        new_weights10 = scores.pop("weights")
        assert pytest.approx(10 * np.mean(new_weights_normed), abs=2) == np.mean(
            new_weights10
        )

    def test_reweight_train_only(self):
        ds1, ds2, ds3 = _create_data()
        scores = reweight_new(
            apply_data=None,
            real=ds2,
            mc=ds3,
            columns=reweight_branches,
            reweighter="gb",
            reweight_cfg=reweight_cfg,
            n_reweights=5,
            add_weights=True,
        )
        weights = reweight_new(
            apply_data=ds1, reweighter=scores["reweighter"], add_weights=True
        )


def _create_data(n_storages=3):
    data_storages = []

    for i in range(n_storages):
        data = pd.DataFrame(
            np.random.normal(0.3 * i, 10 + i, size=[50, 4]), columns=all_branches
        )
        weights = np.random.normal(size=len(data))
        data_storages.append(
            HEPDataStorage(
                data,
                target=i % 2,
                sample_weights=weights,
                data_name="test storage " + str(i),
            )
        )

    return data_storages

    if __name__ == "__main__":
        unittest.main()
