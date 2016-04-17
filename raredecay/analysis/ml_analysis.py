# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: mayou

Module which consist of machine-learning methods to bring useful methods
together into one and use the HEPDataStorage.

It is integrated into the analysis package and depends on the tools.
"""
# debug

import warnings
import memory_profiler

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import hep_ml.reweight
from raredecay.tools import dev_tool, data_tools
from raredecay import globals_

# import the specified config file
# TODO: is this import really necessary? Best would be without config...
import importlib
from raredecay import meta_config
cfg = importlib.import_module(meta_config.run_config)

logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


def reweight_mc_real(reweight_data_mc, reweight_data_real,
                     reweighter='gb', reweight_saveas=None, meta_cfg=None,
                     weights_mc=None, weights_real=None):
    """Return a trained reweighter from a mc/real distribution comparison.

    | Reweighting a distribution is a "making them the same" by changing the \
    weights of the bins (instead of 1) for each event. Mostly, and therefore \
    the naming, you want to change the mc-distribution towards the real one.
    | There are two possibilities

    * normal bins reweighting:
       divides the bins from one distribution by the bins of the other
       distribution. Easy and fast, but unstable and inaccurat for higher
       dimensions.
    * Gradient Boosted reweighting:
       uses several decision trees to reweight the bins. Slower, but more
       accurat. Very useful in higher dimensions.

    Parameters
    ----------
    reweight_data_mc : :class:`HEPDataStorage` (depreceated: root-dict)
        The Monte-Carlo data, which has to be "fitted" to the real data.
        Should be a HEPDataStorage, for compatibility a root-dict is allowed
    reweight_data_real : :class:`HEPDataStorage` (depreceated: root-dict)
        Same as reweight_data_mc but for the real data
    reweighter : {'gb', 'bins'}
        Specify which reweighter to be used
    reweight_saveas : string
        To save a trained reweighter in addition to return it. The value
        is the file(path +)name. The full name will be
         PICKLE_PATH + reweight_saveas + .pickle
        (.pickle is only added if not yet contained in "reweight_saveas")
    meta_cfg : dict
        Contains the parameters for the bins/gb-reweighter. See also
        :func:`~hep_ml.reweight.BinsReweighter` and
        :func:`~hep_ml.reweight.GBReweighter`.
    weights_mc : numpy.array [n_samples]
        Apply weights to the Monte-Carlo data
    weights_real : numpy.array [n_samples]
        Apply weights to the real data.

    Returns
    -------
    out : object of type reweighter
        Reweighter is trained to the data. Can, for example,
        be used with :func:`~hep_ml.reweight.GBReweighter.predict_weights`
    """
    REWEIGHT_MODE = {'gb': 'GB', 'bins': 'Bins', 'bin': 'Bins'}
    try:
        reweighter = REWEIGHT_MODE.get(reweighter.lower())
    except KeyError:
        logger.critical("Reweighter invalid: " + reweighter +
                        ". Probably wrong defined in config.")
        raise ValueError
    reweighter += 'Reweighter'
    # compatibility only!
    original = reweight_data_mc
    target = reweight_data_real
    reweighter = getattr(hep_ml.reweight,
                         reweighter)(**meta_cfg)
    reweighter.fit(original=original.pandasDF(), target=target.pandasDF(),
                   original_weight=original.get_weights(),
                   target_weight=target.get_weights())
    return data_tools.adv_return(reweighter, logger=logger,
                                 save_name=reweight_saveas)


def reweight_weights(reweight_data, reweighter_trained,
                     add_weights_to_data=True):
    """Adds (or only returns) new weights to the data by applying a given
    reweighter on the data.

    Can be seen as a wrapper for the
    :func:`~hep_ml.reweight.GBReweighter.predict_weights` method.
    Additional functionality:
     * Takes a trained reweighter as argument, but can also unpickle one
       from a file.
     * Converts data implicitly to the right format or loads directly if
       already converted to the right format once.

    Parameters
    ----------
    reweight_data : :class:`HEPDataStorage` (depreceated: root-dict)
        The data for which the reweights are to be predicted.
    reweighter_trained : reweighter (*from hep_ml*) or pickle file
        The trained reweighter, which predicts the new weights.
    add_weights_to_data : boolean
        If set to False, the weights will only be returned and not added to
        the data.

    Returns
    ------
    out : numpy.array
        Return a numpy.array of shape [n_samples] containing the new
        weights.
    """
    reweighter_trained = data_tools.try_unpickle(reweighter_trained)
    new_weights = reweighter_trained.predict_weights(reweight_data.pandasDF(),
                                  original_weight=reweight_data.get_weights())
    if add_weights_to_data:
        reweight_data.set_weights(new_weights)
    return new_weights


def draw_distributions(data_to_plot, figure_number, labels=None, weights=None,
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
    data_to_plot, weights = data_tools.format_data_weights(
            data_to_shape=data_to_plot, weights=weights)
    labels = dev_tool.make_list_fill_var(labels, len(data_to_plot),
                                         var=None)
    if columns is None:
        columns = list(data_to_plot[0].columns.values)
    subplot_col = math.ceil(math.sqrt(len(columns)))
    subplot_row = math.ceil(float(len(columns))/subplot_col)
    plt.figure(figure_number)
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


def fast_ROC_AUC(original, target, weight_original=None,
                 weight_target=None, config_clf=None,
                 take_label_from_data=False):
    """ Return the ROC AUC fast, useful to find out, how well they can be
    distinguished.

    Learn to distinguish between monte-carl data (original) and real data
    (target)
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    from sklearn.cross_validation import KFold, cross_val_score
    DEFAULT_CONFIG_CLF= dict(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=7
        )
    if config_clf is None:
        config_clf = {}
    config_clf = dict(DEFAULT_CONFIG_CLF, **config_clf)
    if weight_original is None:
        weight_original = []
    if weight_target is None:
        weight_target = []


    original_data = original.pandasDF()
    target_data = target.pandasDF()
    data = pd.concat([original_data, target_data])
    weights = np.concatenate((original.get_weights(), target.get_weights()))
    # maybe useful? I don't think
    if take_label_from_data:
        label = np.concatenate((original.get_targets(), target.get_targets()))
    else:
        label = np.array([0] * len(original) + [1] * len(target))
    # assertions no more required?
    assert len(weight_original) in (0, len(original)), "weights and data have different lengts"
    assert len(weight_target) in (0, len(target)), "weights and data have different lengts"

    # first way of getting roc auc score
    X_train, X_test, y_train, y_test, weight_train, weight_test = (
        train_test_split(data, label, weights, test_size=0.5,
                         random_state=globals_.randint))
    clf = GradientBoostingClassifier(random_state=globals_.randint+1, **config_clf)
    plt.figure('training1 dataset')
    plt.scatter(X_train['B_PT'], X_train['nTracks'], label='training', alpha=0.3)
    plt.scatter(X_test['B_PT'], X_test['nTracks'], color='r', label='test', alpha=0.3)
    plt.legend()
    # test end
    clf.fit(X_train, y_train, weight_train)
    ROC_AUC = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1],
                            sample_weight=weight_test)

    # second way of getting roc
    X_train, X_test, y_train, y_test, weight_train, weight_test = (
        train_test_split(data, label, weights,
                         random_state=globals_.randint))

    plt.figure('training2 dataset')
    plt.scatter(X_train['B_PT'], X_train['nTracks'], label='training', alpha=0.3)
    plt.scatter(X_test['B_PT'], X_test['nTracks'], color='r', label='test', alpha=0.3)
    plt.legend()

    clf = GradientBoostingClassifier(random_state=globals_.randint+1, **config_clf)
    clf.fit(X_train, y_train, weight_train)
#    y_score = clf.predict_proba(X_test)[:, 1]
    y_score = clf.predict_proba(X_test)[:, 1]
    logger.debug("predict_proba: " + str(y_score))
    plt.figure(('prediction probabilities' + str(sum(weight_train))))
    plt.hist(y_score, bins=50)
    fpr, tpr, temp = roc_curve(y_test, y_score, sample_weight=weight_test)
    logger.debug("fpr: " + str(fpr) + "\n\ntpr: " + str(tpr))
    roc_auc = auc(fpr, tpr, reorder=True)
    return [ROC_AUC, roc_auc, fpr, tpr]


def fast_to_pandas(data_in, **kwarg_to_pandas):
    """ Check if data has already been converted and saved before calling
    to_pandas.

    "Better" version of :func:`~data_tools.to_pandas` and
    identical to it if :func:`~config.FAST_CONVERSION` is set to False.
    """
    raise RuntimeError("This function is not implemented, as it is obsolet")
    add_to_already_pandas = False
    if cfg.FAST_CONVERSION:
#     @todo   dic, data_in = next((c for c in already_pandas if
 #                           data_in == c[0]), (None, data_in))
        if dic is None:
            dictionary = dict(data_in)
            add_to_already_pandas = True
    data_in = data_tools.to_pandas(data_in, logger, **kwarg_to_pandas)
    if add_to_already_pandas:
        already_pandas.append((dictionary, data_in))
    return data_in

    logger.info("module finished")
