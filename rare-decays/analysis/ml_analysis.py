# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: mayou
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import hep_ml.reweight
from tools import dev_tool, data_tools
import config as cfg


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

        self.logger.debug("starting data conversion")
        original = self.fast_to_pandas(reweight_data_mc)
        target = self.fast_to_pandas(reweight_data_real)
        self.logger.debug("data converted to pandas")
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
                           columns=None, hist_cfg=cfg.hist_cfg_std, show=False,
                           multithread=cfg.MULTITHREAD):
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
        data_to_plot = dev_tool.make_list_fill_none(data_to_plot)
        labels = dev_tool.make_list_fill_none(labels, len(data_to_plot))
        weights = dev_tool.make_list_fill_none(weights, len(data_to_plot))
        self.logger.debug("data_to_plot: " + str(data_to_plot))
        self.logger.debug("labels: " + str(labels))
        self.logger.debug("weights: " + str(weights))
        if multithread:
            pass
        else:
            data_to_plot = map(self.fast_to_pandas, data_to_plot)
        if columns is None:
            columns = list(data_to_plot[0].columns.values)
            self.logger.debug("columns: " + str(columns))
        subplot_col = math.ceil(math.sqrt(len(data_to_plot)-0.001))
        subplot_row = math.ceil(float(len(data_to_plot))/subplot_col)
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