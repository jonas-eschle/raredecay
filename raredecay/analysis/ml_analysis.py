# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: mayou
"""


import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import hep_ml.reweight
from raredecay.tools import dev_tool, data_tools
#import config as cfg
import importlib
from raredecay import meta_config
cfg = importlib.import_module(meta_config.run_config)


class MachineLearningAnalysis:
    """Use ML-techniques on datasets to reweight, train and classify


    """
    already_pandas = []
    __figure_number = 0
    __REWEIGHT_MODE = {'gb': 'GB', 'bins': 'Bins'}  # for GB/BinsReweighter
    __REWEIGHT_MODE_DEFAULT = 'gb'  # user-readable

    def __init__(self):
        self.logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)

    def reweight_mc_real(self, reweight_data_mc, reweight_data_real,
                         reweighter='gb', weights_mc=None, weights_real=None,
                         reweight_saveas=None, meta_cfg=None):
        """Return a trained reweighter from a mc/real distribution comparison.

        Reweighting a distribution is a "making them the same" by changing the
        weights of the bins. Mostly, and therefore the naming, you want to
        change the mc-distribution towards the real one.
        There are two possibilities

        * normal bins reweighting:
           divides the bins from one distribution by the bins of the other
           distribution. Easy and fast, but unstable and inaccurat for higher
           dimensions.
        * Gradient Boosted reweighting:
           uses several decision trees to reweight the bins. Slower, but more
           accurat. Very useful in higher dimensions.

        Parameters
        ----------
        reweight_data_mc : dict or numpy.ndarray or pandas.DataFrame
            The Monte-Carlo data, which has to be "fitted" to the real data.
            Can be a dictionary containing the rootfile, tree, branch etc. or
            a numpy array or pandas DataFrame, which already hold the data.
        reweight_data_real : dict or numpy.ndarray or pandas.DataFrame
            Same as reweight_data_mc but for the real data
        reweighter : {'gb', 'bins'}
            Specify which reweighter to be used
        weights_mc : numpy.array [n_samples]
            Apply weights to the Monte-Carlo data
        weights_real : numpy.array [n_samples]
            Apply weights to the real data
        reweight_saveas : string
            To save a trained reweighter in addition to return it. The value
            is the file(path +)name. The full name will be
             PICKLE_PATH + reweight_saveas + .pickle
            (.pickle is only added if not yet contained in "reweight_saveas")
        meta_cfg : dict
            Contains the parameters for the bins/gb-reweighter. See also
            :func:`~hep_ml.reweight.BinsReweighter` and
            :func:`~hep_ml.reweight.GBReweighter`.

        Returns
        -------
        out: object of type reweighter
            Reweighter is trained to the data. Can, for example,
            be used with :func:`~hep_ml.reweight.GBReweighter.predict_weights`
        """
        try:
            reweighter = self.__REWEIGHT_MODE.get(reweighter)
        except KeyError:
            self.logger.critical("Reweighter invalid: " + reweighter +
                                 ". Probably wrong defined in config.")
            raise ValueError
        else:
            reweighter += 'Reweighter'
        original = self.fast_to_pandas(reweight_data_mc)
        target = self.fast_to_pandas(reweight_data_real)
        reweighter = getattr(hep_ml.reweight,
                             reweighter)(**meta_cfg)
        reweighter.fit(original, target)
        return data_tools.adv_return(reweighter, self.logger,
                                     save_name=reweight_saveas)

    def reweight_weights(self, reweight_apply_data, reweighter_trained):
        """Return the new weights by applying a given reweighter on the data.

        Can be seen as a wrapper for the
        :func:`~hep_ml.reweight.GBReweighter.predict_weights` method.
        Additional functionality:
         * Takes a trained reweighter as argument, but can also unpickle one
           from a file.
         * Converts data implicitly to the right format or loads directly if
           already converted to the right format once.

        Parameters
        ----------
        reweight_apply_data : dict (*rootfile*) or numpy.array or \
        pandas.DataFrame
            The data for which the reweights are be predicted.
        reweighter_trained : reweighter (*from hep_ml*) or pickle file
            The trained reweighter, which predicts the new weights.

        Returns
        ------
        out : numpy.array
            Return a numpy.array of shape [n_samples] containing the new
            weights.
        """
        reweighter_trained = data_tools.try_unpickle(reweighter_trained)
        reweight_apply_data = self.fast_to_pandas(reweight_apply_data)
        new_weights = reweighter_trained.predict_weights(reweight_apply_data)
        return new_weights

    def draw_distributions(self, data_to_plot, labels=None, weights=None,
                           columns=None, hist_cfg=cfg.hist_cfg_std,
                           show=False):
        """Draw histograms of weighted distributions.


        Parameters
        ----------
        data_to_plot : list with data [numpy.array or tree-dict or
        pandas.DataFrame]
            Distributions to plot.
        weights : list (!) with weights
        [[1-D list containing weights],[weights],[weights],...]
            Specify the weights in the right order for the distributions.

        """
        data_to_plot, weights = data_tools.format_data_weights(data_to_shape=data_to_plot, weights=weights,
                                                         ml_analysis_object=self)
        labels = dev_tool.make_list_fill_var(labels, len(data_to_plot),
                                             var=None)
        if columns is None:
            columns = list(data_to_plot[0].columns.values)
        subplot_col = math.ceil(math.sqrt(len(columns)))
        subplot_row = math.ceil(float(len(columns))/subplot_col)
        self.__figure_number += 1
        plt.figure(self.__figure_number)
        for col_id, column in enumerate(columns, 1):
            x_limits = np.percentile(np.hstack(data_to_plot[0][column]),
                                     [0.01, 99.99])
            plt.subplot(subplot_row, subplot_col, col_id)
            for data_id, data in enumerate(data_to_plot):
                plt.hist(data[column], weights=weights[data_id],
                         range=x_limits, label=labels[data_id], **hist_cfg)
            plt.title(column)
            plt.legend()
        if show:
            plt.show()

    def fast_ROC_AUC(self, original, target, weight_original=None,
                     weight_target=None):
        """ Return the ROC AUC fast, useful to find out, how well they can be
        distinguished.
        """
        if weight_original is None:
            weight_original = []
        if weight_target is None:
            weight_target = []
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.cross_validation import train_test_split
        from sklearn.metrics import roc_auc_score
        from sklearn.cross_validation import KFold, cross_val_score

        original = self.fast_to_pandas(original)
        target = self.fast_to_pandas(target)
        data = pd.concat([original, target])
        label = np.array([0] * len(original) + [1] * len(target))
        assert len(weight_original) in (0, len(original)), "weights and data have different lengts"
        assert len(weight_target) in (0, len(target)), "weights and data have different lengts"
        weight_original = np.array(dev_tool.fill_list_var(weight_original,
                                                 len(original), 1))
        weight_target = np.array(dev_tool.fill_list_var(weight_target,
                                               len(target), 1))
        weights = np.concatenate([weight_original, weight_target])
        X_train, X_test, y_train, y_test, weight_train, weight_test = (
            train_test_split(data, label, weights, random_state=42))
        clf = GradientBoostingClassifier(n_estimators=100)
        #scores = cross_val_score(clf, data, label,
        #                         cv=KFold(len(data), n_folds=3, shuffle=True),
        #                         n_jobs=6, scoring = 'roc_auc')
        #roc_auc_kfold = ("roc_auc_score KFold = %f +-%f" % (np.mean(scores), np.std(scores)))
        roc_auc_kfold = 'not implemented'
        # test end
        clf.fit(X_train, y_train, weight_train)

        ROC_AUC = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1],
                                sample_weight=weight_test)
        return ROC_AUC, roc_auc_kfold

    def fast_to_pandas(self, data_in, **kwarg_to_pandas):
        """ Check if data has already been converted and saved before calling
        to_pandas.

        "Better" version of :func:`~data_tools.to_pandas` and
        identical to it if :func:`~config.FAST_CONVERSION` is set to False.
        """
        add_to_already_pandas = False
        if cfg.FAST_CONVERSION:
            dic, data_in = next((c for c in self.already_pandas if
                                data_in == c[0]), (None, data_in))
            if dic is None:
                dictionary = dict(data_in)
                add_to_already_pandas = True
        data_in = data_tools.to_pandas(data_in, self.logger, **kwarg_to_pandas)
        if add_to_already_pandas:
            self.already_pandas.append((dictionary, data_in))
        return data_in

        self.logger.info("module finished")
