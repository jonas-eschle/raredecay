from unittest import TestCase
import unittest

import pandas as pd
import pandas.util.testing as pdtest
import numpy as np
import copy

from raredecay.tools.data_storage import HEPDataStorage


class TestHEPDataStorageMixin(object):
    def setUp(self):
        self.truth_df, self.truth_weights, self.truth_targets = self._make_dataset()
        self.data_for_hepds, self.target_for_hepds, self.weights_for_hepds = self._generate_data(self.truth_df,
                                                  self.truth_weights,
                                                  self.truth_targets)

        self.ds = HEPDataStorage(self.data_for_hepds,target=self.target_for_hepds,
                                 sample_weights=self.weights_for_hepds,
                                 data_name="init", data_name_addition="init")

    def _make_dataset(self):
        data = pd.DataFrame({'a': list(range(9)),
                             'b': list(range(10, 19)),
                             'c': list(range(20, 29)),
                             'd': list(range(30, 39))})
        targets = [1, 0, 1, 0, 1, 0, 1, 0, 1]
        weights = np.array([1, 1, 1, 1, 1, 2, 3, 4, 0.25])
        return data, targets, weights

    def _generate_data(self, data, targets, weights):
        """Return data file ready to be passed into HEPDataStorage and creating file if necessary
        Returns
        -------
        data
        """
        raise NotImplementedError("Has to be overriden by subclass")

    def test_initialization(self):
        pdtest.assert_frame_equal(self.ds.pandasDF(), self.truth_df)

    def test_get_name(self):
        pass

    def test_name(self):
        pass

    def test__get_name(self):
        pass

    def test__set_name(self):
        pass

    def test_data_name(self):
        pass

    def test_data_name_addition(self):
        pass

    def test_fold_name(self):
        pass

    def test_data_type(self):
        pass

    def test_get_index(self):
        pass

    def test_index(self):
        pass

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
        pass

    def test__make_df(self):
        pass

    def test_get_targets(self):
        pass

    def test__get_targets(self):
        pass

    def test_set_targets(self):
        pass

    def test__set_target(self):
        pass

    def test_make_dataset(self):
        pass

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


class TestHEPDataStoragePandasDF(TestHEPDataStorageMixin, TestCase):
    def _generate_data(self, data, targets, weights):
        return copy.deepcopy(data), copy.deepcopy(targets), copy.deepcopy(weights)


if __name__ == '__main__':
    unittest.main()
