from __future__ import absolute_import, print_function, division

import copy


import unittest
from unittest import TestCase

import root_numpy
import rootpy

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.util.testing as pdtest

from raredecay.tools.data_storage import HEPDataStorage


class TestHEPDataStorageMixin(TestCase):
    def setUp(self):
        self._set_truth()

        self.ds = self._create_ds()

        self._set_truth2()

        self.ds2 = self._create_ds2()

    def _create_ds(self):
        return HEPDataStorage(self.data_for_hepds, target=self.target_for_hepds,
                              sample_weights=self.weights_for_hepds,
                              index=self.truth_index,
                              data_name=self.truth_name, data_name_addition=self.truth_name_addition)

    def _create_ds2(self):
        return HEPDataStorage(self.data_for_hepds2, target=self.target_for_hepds2,
                              sample_weights=self.weights_for_hepds2,
                              data_name=self.truth_name2, data_name_addition=self.truth_name_addition2)

    def _set_truth2(self):
        self.truth_df2, self.truth_targets2, self.truth_weights2 = self._make_dataset2()
        self.data_for_hepds2, self.target_for_hepds2, self.weights_for_hepds2 = self._generate_data(
                copy.deepcopy(self.truth_df2),
                copy.deepcopy(self.truth_targets2),
                copy.deepcopy(self.truth_weights2))
        self.truth_weights_normalized2 = self.truth_weights2 / np.average(self.truth_weights2)
        self.truth_name2 = "ds1"
        self.truth_name_addition2 = "ds1add"

    def _set_truth(self):
        self.truth_df, self.truth_targets, self.truth_weights, index = self._make_dataset()
        self.truth_index = copy.deepcopy(index)
        self.data_for_hepds, self.target_for_hepds, self.weights_for_hepds = self._generate_data(
                copy.deepcopy(self.truth_df),
                copy.deepcopy(self.truth_targets),
                copy.deepcopy(self.truth_weights))
        self.truth_weights_normalized = self.truth_weights / np.average(self.truth_weights)
        self.truth_name = "ds1"
        self.truth_name_addition = "ds1add"
        return

    def _make_dataset(self):
        index = [0, 2, 1, 3, 4, 5, 6, 7, 8]
        data = pd.DataFrame({'a': list(range(9)),
                             'b': list(range(10, 19)),
                             'c': list(range(20, 29)),
                             'd': list(range(30, 39))}, index=index)
        targets = [1, 0, 1, 0, 1, 0, 1, 0, 1]
        weights = np.array([1, 1, 1, 1, 1, 2, 3, 4, 0.25])
        index = [0, 2, 1, 3, 4, 5, 6, 7, 8]
        return (copy.deepcopy(obj) for obj in (data, targets, weights, index))

    def _make_dataset2(self):
        data = pd.DataFrame({'a': list(range(200, 203)),
                             'b': list(range(210, 213)),
                             'c': list(range(220, 223)),
                             'd': list(range(230, 233))})
        targets = [1, 1, 0]
        weights = np.array([1.5, 10, 0.3])
        return data, targets, weights

    def _generate_data(self, data, targets, weights):
        """Return data file ready to be passed into HEPDataStorage and creating file if necessary
        OVERRIDE THIS METHOD (do not depend on the default base implementation)
        Returns
        -------
        data
        """
        self.truth_data_type = "df"
        return copy.deepcopy(data), copy.deepcopy(targets), copy.deepcopy(weights)

    def test_initialization(self):
        pdtest.assert_frame_equal(self.ds.pandasDF(), self.truth_df)
        nptest.assert_almost_equal(self.truth_targets, self.ds.get_targets())
        nptest.assert_almost_equal(self.truth_weights, self.ds.weights)
        nptest.assert_almost_equal(self.truth_weights_normalized,
                                   self.ds.get_weights())

        pdtest.assert_frame_equal(self.ds2.pandasDF(), self.truth_df2)
        nptest.assert_almost_equal(self.truth_targets2, self.ds2.get_targets())
        nptest.assert_almost_equal(self.truth_weights2, self.ds2.weights)
        nptest.assert_almost_equal(self.truth_weights_normalized2,
                                   self.ds2.get_weights())

    def test_get_name(self):
        pass

    def test_name(self):
        pass

    def test__get_name(self):
        pass

    def test__set_name(self):
        pass

    def test_data_name(self):
        self.assertEqual(self.truth_name, self.ds.data_name)

    def test_data_name_addition(self):
        pass

    def test_fold_name(self):
        pass

    def test_data_type(self):
        self.assertEqual(self.ds.data_type, self.truth_data_type)

    def test_get_index(self):
        pass

    def test_index(self):
        nptest.assert_almost_equal(self.truth_index, self.ds.index)

    def test__make_index(self):
        pass

    def test__set_index(self):
        pass

    def test_columns(self):
        pass

    def test__set_columns(self):
        pass

    def test__set_length(self):
        pass

    def test__get_data_type(self):
        pass

    def test_data(self):
        pass

    def test_set_data(self):
        pass

    def test__set_data(self):
        pass

    def test_get_weights(self):
        pass

    def test__get_weights(self):
        pass

    def test_set_weights(self):
        pass

    def test__set_weights(self):
        pass

    def test_set_root_selection(self):
        pass

    def test_pandasDF(self):
        pdtest.assert_almost_equal(self.truth_df, self.ds.pandasDF())
        for cols in (['a', 'b'], ['a', 'b', 'c', 'd']):
            pdtest.assert_almost_equal(self.truth_df[cols], self.ds.pandasDF(columns=cols))
        index = [1, 2, 5]
        indexed_df = pd.DataFrame({'a': [2, 1, 5],
                                   'b': [12, 11, 15],
                                   'c': [22, 21, 25],
                                   'd': [32, 31, 35]}, index=index)
        pdtest.assert_frame_equal(indexed_df, self.ds.pandasDF(index=index))

    def test__make_df(self):
        pass

    def test_get_targets(self):
        index = [1, 2, 5]
        nptest.assert_almost_equal([1, 0, 0], self.ds.get_targets(index=index))

    def test__get_targets(self):
        pass

    def test_set_targets(self):
        pass

    def test__set_target(self):
        pass

    def test_make_dataset(self):
        data, targets, weights = self.ds.make_dataset(second_storage=self.ds2, weights_ratio=0)
        truth_data = pd.concat((self.truth_df, self.truth_df2), axis=0, ignore_index=True)
        truth_targets = np.concatenate((self.truth_targets, self.truth_targets2))
        truth_weights = np.concatenate((self.truth_weights, self.truth_weights2))

        pdtest.assert_almost_equal(truth_data, data)
        nptest.assert_almost_equal(truth_targets, targets)
        nptest.assert_almost_equal(truth_weights, weights)

    def test_copy_storage(self):
        pass

    def test_get_LabeledDataStorage(self):
        pass

    def test_make_folds(self):
        pass

    def test_get_fold(self):
        pass

    def test_get_n_folds(self):
        pass

    def test_plot(self):
        pass

    def tearDown(self):

        self._tearDown()

    def _tearDown(self):
        pass


class TestHEPDataStoragePandasDF(TestHEPDataStorageMixin, TestCase):
    def _generate_data(self, data, targets, weights):
        self.truth_data_type = "df"
        return copy.deepcopy(data), copy.deepcopy(targets), copy.deepcopy(weights)


class TestHEPDataStorageROOT(TestHEPDataStorageMixin, TestCase):

    def _generate_data(self, data, targets, weights):

        from root_pandas import to_root

        import tempfile

        data['root_w'] = weights
        tmp_file = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
        self.temp_file_path_root = tmp_file.name
        to_root(data, tmp_file.name, key='DecayTree')
        self.truth_data_type = "root"

        root_dict = {'filenames': tmp_file.name,
                     'treename': 'DecayTree',
                     'branches': ['a', 'b', 'c', 'd']}


        return root_dict, copy.deepcopy(targets), 'root_w'

        def _create_ds(self):
            return HEPDataStorage(self.data_for_hepds, target=self.target_for_hepds,
                                  sample_weights=self.weights_for_hepds,
                                  index=self.truth_index,  # NO index, because it is saved sorted
                                  data_name=self.truth_name, data_name_addition=self.truth_name_addition)




        return
    def _tearDown(self):
        import os
        os.remove(self.temp_file_path_root)


if __name__ == '__main__':
    unittest.main()
