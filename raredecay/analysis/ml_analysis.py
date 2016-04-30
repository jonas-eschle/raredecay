# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: mayou

Module which consist of machine-learning methods to bring useful methods
together into one and use the HEPDataStorage.

It is integrated into the analysis package and depends on the tools.
"""
from __future__ import division, absolute_import

import warnings
import memory_profiler

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import hep_ml.reweight
from raredecay.tools import dev_tool, data_tools
from raredecay import globals_

# import the specified config file
# TODO: is this import really necessary? Best would be without config...
import importlib
from raredecay import meta_config
cfg = importlib.import_module(meta_config.run_config)
logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


def reweight_mc_real(reweight_data_mc, reweight_data_real, branches=None,
                     reweighter='gb', reweight_saveas=None, meta_cfg=None,
                     weights_mc=None, weights_real=None):
    """Return a trained reweighter from a (mc/real) distribution comparison.

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
    reweight_data_mc : :class:`HEPDataStorage`
        The Monte-Carlo data, which has to be "fitted" to the real data.
    reweight_data_real : :class:`HEPDataStorage` (depreceated: root-dict)
        Same as *reweight_data_mc* but for the real data.
    branches : list of strings
        The columns/features/branches you want to use for the reweighting.
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
        Explicit weights for the Monte-Carlo data. Only specify if you don't
        want to use the weights in the *HEPDataStorage*.
    weights_real : numpy.array [n_samples]
        Explicit weights for the real data. Only specify if you don't
        want to use the weights in the *HEPDataStorage*.

    Returns
    -------
    out : object of type reweighter
        Reweighter is trained to the data. Can, for example,
        be used with :func:`~hep_ml.reweight.GBReweighter.predict_weights`
    """
    __REWEIGHT_MODE = {'gb': 'GB', 'bins': 'Bins', 'bin': 'Bins'}

    # check for valid user input
    try:
        reweighter = __REWEIGHT_MODE.get(reweighter.lower())
    except KeyError:
        raise ValueError("Reweighter invalid: " + reweighter)
    reweighter += 'Reweighter'
    logger.info("Reweighter: " + str(reweighter) + " with config: " + str(meta_cfg))
    reweighter = getattr(hep_ml.reweight, reweighter)(**meta_cfg)

    # do the reweighting
    reweighter.fit(original=reweight_data_mc.pandasDF(branches=branches),
                   target=reweight_data_real.pandasDF(branches=branches),
                   original_weight=reweight_data_mc.get_weights(),
                   target_weight=reweight_data_real.get_weights())
    return data_tools.adv_return(reweighter, logger=logger,
                                 save_name=reweight_saveas)


def reweight_weights(reweight_data, reweighter_trained, branches=None,
                     normalize=True, add_weights_to_data=True):
    """Adds (or only returns) new weights to the data by applying a given
    reweighter on the data.

    Can be seen as a wrapper for the
    :func:`~hep_ml.reweight.GBReweighter.predict_weights` method.
    Additional functionality:
     * Takes a trained reweighter as argument, but can also unpickle one
       from a file.

    Parameters
    ----------
    reweight_data : :class:`HEPDataStorage`
        The data for which the reweights are to be predicted.
    reweighter_trained : (pickled) reweighter (*from hep_ml*)
        The trained reweighter, which predicts the new weights.
    normalize : boolean
        If True, the weights will be normalized to one.
    add_weights_to_data : boolean
        If set to False, the weights will only be returned and not updated in
        the data (*HEPDataStorage*).

    Returns
    ------
    out : numpy.array
        Return a numpy.array of shape [n_samples] containing the new
        weights.
    """
    reweighter_trained = data_tools.try_unpickle(reweighter_trained)
    new_weights = reweighter_trained.predict_weights(reweight_data.pandasDF(branches=branches),
                                        original_weight=reweight_data.get_weights())
    if normalize:
        new_weights *= new_weights.size/new_weights.sum()
    if add_weights_to_data:
        reweight_data.set_weights(new_weights)
    return new_weights


def data_ROC(original_data, target_data, plot=True, curve_name=None, n_folds=1,
             weight_original=None, weight_target=None, config_clf=None,
             take_target_from_data=False, use_factory=True):
    """ Return the ROC AUC; useful to find out, how well two datasets can be
    distinguished.

    Learn to distinguish between monte-carl data (original) and real data
    (target) and report (plot) the ROC and the AUC.

    Parameters
    ----------
    original_data : instance of :class:`HEPDataStorage`
        The original or monte-carlo data
    target_data : instance of :class:`HEPDataStorage`
        The target or real data
    plot : boolean
        If true, ROC is plotted. Otherwise, only the ROC AUC is calculated.
    curve_name : str
        Name to label the plottet ROC curve.
    n_folds : int
        Specify how many folds and checks should be made for the training/test.
        If it is 1, a normal trait-test-split with 2/3 - 1/3 ratio is done.
    weight_original : numpy array 1-D [n_samples]
        The weights for the original data. Only use if you don't want to use
        the weights contained in the original_data.
    weight_target : numpy array 1-D [n_samples]
        The weights for the target data. Only use if you don't want to use
        the weights contained in the target_data.
    config_clf : dict
        The configuration for the classifier. If None, a default config is
        taken.
    take_target_from_data : boolean
        If true, the target labeling (say what is original resp. target) is
        taken from data instead of assigned.
        So the name "original_data" and "target_data" has "no meaning" anymore.

    Returns
    -------
    out : float
        The ROC AUC from the classifier on the test samples.
    """
    __DEFAULT_CONFIG_CLF = dict(
        n_estimators=400,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.9,
        max_features=None
    )

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
    from rep.estimators import SklearnClassifier, XGBoostClassifier, TMVAClassifier
    from rep.utils import train_test_split
    from rep.report.metrics import RocAuc
    from sklearn import tree
    from rep.metaml import ClassifiersFactory
    import rep


    config_clf = {} if config_clf is None else config_clf
    config_clf = dict(__DEFAULT_CONFIG_CLF, **config_clf)
    curve_name = 'data' if curve_name is None else curve_name

    # concatenate the original and target data
    data = pd.concat([original_data.pandasDF(), target_data.pandasDF()])
    # take weights from data if not explicitly specified
    if dev_tool.is_in_primitive(weight_original, None):
        weight_original = original_data.get_weights()
    if dev_tool.is_in_primitive(weight_target, None):
        weight_target = target_data.get_weights()
    assert len(weight_original) == len(original_data), "Original weights have wrong length"
    assert len(weight_target) == len(target_data), "Target weights have wrong length"
    weights = np.concatenate((weight_original, weight_target))

    if take_target_from_data:  # if "original" and "target" are "mixed"
        label = np.concatenate((original_data.get_targets(),
                                target_data.get_targets()))
    else:
        label = np.array([0] * len(original_data) + [1] * len(target_data))

    # start ml-part
    clf = SklearnClassifier(GradientBoostingClassifier(
                                random_state=globals_.randint+5, **config_clf))
    # getting roc (auc score) for 1 fold
    if n_folds == 1:
        X_train, X_test, y_train, y_test, weight_train, weight_test = (
            train_test_split(data, label, weights, test_size=0.33,
                             random_state=globals_.randint))
        if use_factory:
            clf_xgb = XGBoostClassifier(n_estimators=1000, eta=0.1, nthreads=8, max_depth=8)
            clf_rnd_forest = SklearnClassifier(RandomForestClassifier(n_estimators=1000, n_jobs=-1))
            clf_ada_xgb = SklearnClassifier(AdaBoostClassifier(base_estimator=XGBoostClassifier(n_estimators=20, eta=0.1), n_estimators=20 ,learning_rate=0.7))
            clf_ada_forest = SklearnClassifier(AdaBoostClassifier(n_estimators=1000, learning_rate=0.05))
            clf_tmva = TMVAClassifier()
            clf_gb = SklearnClassifier(GradientBoostingClassifier(random_state=globals_.randint+5, **config_clf))
            factory = ClassifiersFactory()
            factory.add_classifier('Gradient Boosting', clf_gb)
            #factory.add_classifier('tmva', clf_tmva)
            factory.add_classifier('XGBoost', clf_xgb)
            factory.add_classifier('random forest', clf_rnd_forest)
            #factory.add_classifier('AdaBoost over XGBoost', clf_ada_xgb)
            factory.add_classifier('AdaBoost over random forest', clf_ada_forest)
            clf = factory
            clf.fit(X_train, y_train, weight_train, parallel_profile='threads-4')

        else:
            clf = XGBoostClassifier(n_estimators=200, eta=0.1, nthreads=8, max_depth=8)
            #clf = TMVAClassifier(NTrees=150, Shrinkage=0.8, AdaBoostBeta=0.3)
            #clf = SklearnClassifier(RandomForestClassifier(n_estimators=50, n_jobs=-1))
            #clf = SklearnClassifier(AdaBoostClassifier(base_estimator=XGBoostClassifier(n_estimators=20, eta=0.2), n_estimators=70 ,learning_rate=0.7))
            #clf = SklearnClassifier(AdaBoostClassifier(n_estimators=300, learning_rate=0.05))
            #clf = SklearnClassifier(GradientBoostingClassifier(random_state=globals_.randint+5, **config_clf))
            clf.fit(X_train, y_train, weight_train)
# FIXME:
        #plt.figure()


        report = clf.test_on(X_test, y_test, weight_test)
    else:
        # TODO: maybe implement for more then 1 fold
        raise NotImplementedError("n_folds >1 not yet implemented. Sorry!")

    plt.figure("Learning curve of classifier")
    report.learning_curve(RocAuc(), steps=1).plot(new_plot=True, title="Learning curve of classifiers")
    if plot and use_factory:
        ROC_AUC = report.compute_metric(RocAuc())
        print "Roc auc = ", ROC_AUC
        try:
            report.feature_importance().plot(new_plot=True)
        except:
            warnings.warn("feature importance not calculated due to (most probably) runtime error", RuntimeWarning)
        report.feature_importance_shuffling().plot(new_plot=True)
        report.features_correlation_matrix_by_class().plot(new_plot=True)
        report.features_pdf().plot(new_plot=True)
        #report.metrics_vs_cut(rep.report.metrics.RocAuc).plot(new_plot=True)  # , metric_label="ROC AUC"
        report.prediction_pdf().plot(new_plot=True)

        # TODO: change title because it plots now only one ROC, use labels
        title = ("ROC curve for comparison of " + original_data.get_name() +
                 " and " + target_data.get_name() + "\nROC AUC: " + str(ROC_AUC))
        # curve_name += "AUC = " + str(round(ROC_AUC, 3))
        plt.figure("Data reweighter comparison")
        #report.prediction[curve_name] = report.prediction.pop('clf')
        report.roc(physical_notion=False).plot(new_plot=True, title=title)
        plt.plot([0, 1], [0, 1], 'k--')
    elif plot:
        ROC_AUC = report.compute_metric(RocAuc())['clf']
        title = ("ROC curve for comparison of " + original_data.get_name() +
                 " and " + target_data.get_name())
        curve_name += "AUC = " + str(round(ROC_AUC, 3))
        plt.figure("Data reweighter comparison")
        report.prediction[curve_name] = report.prediction.pop('clf')
        report.roc(physical_notion=False).plot(new_plot=False, title=title)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.figure()
        proba = clf.predict_proba(X_test)[:, 1]
        plt.hist(proba, bins=20, histtype='bar')
    return ROC_AUC
