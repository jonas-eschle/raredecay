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
import multiprocessing
import copy

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter

import hep_ml.reweight
from rep.metaml import ClassifiersFactory
from rep.utils import train_test_split
from rep.data import LabeledDataStorage

# classifier imports
from rep.estimators import SklearnClassifier, XGBoostClassifier, TMVAClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from rep.estimators.theanets import TheanetsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# scoring and validation
from rep.metaml.folding import FoldingClassifier
from rep.report import metrics
from rep.report.classification import ClassificationReport
from sklearn.metrics import recall_score, accuracy_score, classification_report

# Hyperparameter optimization
from rep.metaml import GridOptimalSearchCV, FoldingScorer, RandomParameterOptimizer, SubgridParameterOptimizer
from rep.metaml.gridsearch import RegressionParameterOptimizer, AnnealingParameterOptimizer

from raredecay.tools import dev_tool, data_tools, data_storage
from raredecay import globals_
from raredecay.globals_ import out

# import configuration
import importlib
from raredecay import meta_config
cfg = importlib.import_module(meta_config.run_config)
logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)



def _make_data(original_data, target_data=None, features=None, target_from_data=False,
                  weight_original=None, weight_target=None, conv_ori_weights=False,
                  conv_tar_weights=False, weights_ratio=0):
    """Return the concatenated data, weights and labels for classifier training
    """
# TODO: remove lower part
    data_out = original_data.make_dataset(target_data, columns=features, weights_as_events=conv_ori_weights,
                                          weights_as_events_2=conv_tar_weights, weights_ratio=weights_ratio)

    return data_out

    min_weight = None
    if (original_data is not None) and (target_data is not None) and conv_ori_weights >= 1 and conv_tar_weights >= 1:
        if dev_tool.is_in_primitive(weight_original, None):
            weight_original = original_data.get_weights()
        if dev_tool.is_in_primitive(weight_target, None):
            weight_target = target_data.get_weights()
        min_weight = min(min(weight_original), min(weight_target))

    if dev_tool.is_in_primitive(weight_original, None) or conv_ori_weights >= 1:
            weight_original = original_data.get_weights(weights_as_events=conv_ori_weights, min_weight=min_weight)
    #assert len(weight_original) == len(original_data), "Original weights have wrong length"

    if target_data is None:
        data = original_data.pandasDF(columns=features, weights_as_events=conv_ori_weights, min_weight=min_weight)
        weights = weight_original
        label = original_data.get_targets(weights_as_events=conv_ori_weights, min_weight=min_weight)
    else:
        # concatenate the original and target data
        original = original_data.pandasDF(columns=features, weights_as_events=conv_ori_weights, min_weight=min_weight)
        target = target_data.pandasDF(columns=features, weights_as_events=conv_tar_weights, min_weight=min_weight)
        data = pd.concat([original, target])

        # take weights from data if not explicitly specified
        if dev_tool.is_in_primitive(weight_target, None) or conv_ori_weights >= 1:
            weight_target = target_data.get_weights(weights_as_events=conv_tar_weights, min_weight=min_weight)
        #assert len(weight_target) == len(target_data), "Target weights have wrong length"
        weights = np.concatenate((weight_original, weight_target))

        if target_from_data:  # if "original" and "target" are "mixed"
            label = np.concatenate((original_data.get_targets(weights_as_events=conv_ori_weights, min_weight=min_weight),
                                    target_data.get_targets(weights_as_events=conv_tar_weights, min_weight=min_weight)))
        else:
            label = np.concatenate((np.zeros(len(original)), np.ones(len(target))))

    return data, label, weights


def optimize_hyper_parameters(original_data, target_data, clf, config_clf,
                              features=None, optimize_features=False,
                              take_target_from_data=False, train_best=False):
    """Optimize the hyperparameters of a classifier"""

    # initialize variables and setting defaults
    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    save_ext_fig_cfg = dict(meta_config.DEFAULT_EXT_SAVE_FIG, **cfg.save_ext_fig_cfg)
    n_eval = cfg.hyper_cfg['n_evaluations']
    n_checks = cfg.hyper_cfg['n_fold_checks']
    n_folds = cfg.hyper_cfg['n_folds']
    generator_type = cfg.hyper_cfg.get('generator', meta_config.DEFAULT_HYPER_GENERATOR)
    parallel_profile = 'threads-' + str(min(globals_.free_cpus(), n_eval))

    # Create parameter for clf and hyper-search
    grid_param = {}
    list_param = ['layers', 'trainers']  # parameters which are by their nature a list, like nn-layers
    for key, val in config_clf.items():
        if isinstance(val, (list, np.ndarray, pd.Series)):
            if key not in list_param or isinstance(val[0], list):
                val = data_tools.to_list(val)
                grid_param[key] = config_clf.pop(key)

    assert grid_param != {}, "No values for optimization found"

    # rederict print output (from hyperparameter-optimizer)
    out.IO_to_string()

    # initialize data
    data, label, weights = _make_data(original_data, target_data, features=features,
                                      target_from_data=take_target_from_data)

    # initialize classifier
    if clf == 'xgb':
        clf_name = "XGBoost"
        config_clf.update(nthreads=1)
        clf = XGBoostClassifier(**config_clf)
    elif clf == 'rdf':
        clf_name = "Random Forest"
        config_clf.update(n_jobs=globals_.free_cpus())  # needs less RAM if parallelized this way
        parallel_profile=None
        clf = SklearnClassifier(RandomForestClassifier(**config_clf))
    elif clf == 'gb':
        clf_name = "GradientBoosting classifier"
        clf = SklearnClassifier(GradientBoostingClassifier(**config_clf))
    elif clf == 'nn':
        clf_name = "Theanets Neural Network"
        parallel_profile=None if meta_config.use_gpu else parallel_profile
        clf = TheanetsClassifier(**config_clf)
    elif clf == 'erf':
        clf_name = "Extra Random Forest"
        clf = SklearnClassifier(ExtraTreesClassifier(**config_clf))
    elif clf == 'ada':
        clf_name = "AdaBoost classifier"
        clf = SklearnClassifier(AdaBoostClassifier(**config_clf))


    if generator_type == 'regression':
        generator = RegressionParameterOptimizer(grid_param, n_evaluations=n_eval)
    elif generator_type == 'subgrid':
        generator = SubgridParameterOptimizer(grid_param, n_evaluations=n_eval)
    elif generator_type == 'random':
        generator = RandomParameterOptimizer(grid_param, n_evaluations=n_eval)
    else:
        raise ValueError(str(generator) + " not a valid, implemented generator")
    scorer = FoldingScorer(metrics.RocAuc(), folds=n_folds, fold_checks=n_checks)
    grid_finder = GridOptimalSearchCV(clf, generator, scorer, parallel_profile=parallel_profile)

    # Search for hyperparameters
    logger.info("starting " + clf_name + " hyper optimization")
    grid_finder.fit(data, label, weights)
    logger.info(clf_name + " hyper optimization finished")
    grid_finder.params_generator.print_results()


    if train_best:
        # Train the best and plot reports
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(data, label, weights,
                                                                             test_size=0.25)
        best_est = grid_finder.fit_best_estimator(X_train, y_train, w_train)
        report = best_est.test_on(X_test, y_test, w_test)

        # plots
        try:
            name1 = "Learning curve of best classifier"
            out.save_fig(name1, **save_fig_cfg)
            report.roc().plot(title=name1)
            name2 = str("Feature correlation matrix of " + original_data.get_name() +
                        " and " + target_data.get_name())
            out.save_fig(name2, **save_ext_fig_cfg)
            report.features_correlation_matrix().plot(title=name2)
            name3 = "Learning curve of best classifier"
            out.save_fig(name3, **save_fig_cfg)
            report.learning_curve(metrics.RocAuc(), steps=1).plot(title=name3)
            name4 = "Feature importance of best classifier"
            out.save_fig(name4, **save_fig_cfg)
            report.learning_curve(metrics.RocAuc(), steps=1).plot(title=name4)
        except:
            logger.error("Could not plot hyper optimization " + clf_name + " plots")

    out.IO_to_sys(subtitle="XGBoost hyperparameter optimization")


def classify(original_data=None, target_data=None, features=None, validation=10, clf='xgb',
             make_plots=True, plot_title=None, curve_name=None, target_from_data=False,
             conv_ori_weights=False, conv_tar_weights=False, conv_vali_weights=False):
    """Training and testing a classifier or distinguish a dataset

    Parameters
    ----------
    original_data : HEPDataStorage
        The original data for the training
    target_data : HEPDataStorage or None
        The target data for the training. If None, only the original_data will
        be used for the training.
    features : list(str, str, str...)
        List with features/columns to use in training.
    validation : int >= 1 or HEPDataStorage
        You can either do cross-validation or give a testsample for the data.

        * Cross-validation:
            Enter an integer, which is the number of folds
        * Validation-dataset:
            Enter a *HEPDataStorage* which contains data to be tested on.
            The target-label will be taken from it, so ensure that they are
            not None! To use two datasets, you can also use a list of
            **maximum** two datastorages.
    clf : str {'xgb'} or rep-classifier
        The classifier to be used for the training and predicting. If you don't
        pass a classifier (with *fit*, *predict* and *predict_proba* at least),
        an XGBoost classifier will be used.
    make_plots : boolean
        If True, plots of the classification will be made.
    plot_title : str
        A part of the title of the plots.
    curve_name : str
        A labeling for the plotted data.
    target_from_data : boolean
        | If true, the target-values (labels; 0 or 1) for the original and the
          target data will be taken from the
          data instead of assigned accordingly (original:1, target:0).
        | If no target_data is provided, the targets/labels will be taken
          from the original_data anyway.

    Returns
    -------
    out : clf
        Return the trained classifier.
    .. note::
        If validation was choosen to be KFold, the returned classifier well be
        of instance :py:class:`~rep.metaml.folding.FoldingClassifier()`!
    out : float (only if validation is not None)
        Return the score (recall or roc auc) of the validation.
    """
    logger.info("Starting classify with " + str(clf))
    # initialize variables and data
    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    save_ext_fig_cfg = dict(meta_config.DEFAULT_EXT_SAVE_FIG, **cfg.save_ext_fig_cfg)

    plot_title = "classify" if plot_title is None else plot_title
    if (original_data is None) and (target_data is not None):
        original_data, target_data = target_data, original_data  # switch places
    if original_data is not None:
        data, label, weights = _make_data(original_data, target_data, features=features,
                                                       conv_ori_weights=conv_ori_weights,
                                                       conv_tar_weights=conv_tar_weights)
        data_name = original_data.get_name()
        if target_data is not None:
            data_name += " and " + target_data.get_name()
    clf_name = 'classifier'
    if clf == 'xgb':
        clf_name = 'XGBoost classifier'
        clf = XGBoostClassifier(**dict(meta_config.DEFAULT_CLF_XGB, nthreads=globals_.free_cpus()))
        is_parallel = True
    if clf == 'rdf':
        clf_name = "RandomForest classifier"
        cfg_clf = meta_config.DEFAULT_CLF_RDF,
        cfg_clf = dict(n_jobs=globals_.free_cpus(), random_state=globals_.randint+432)
        clf = SklearnClassifier(RandomForestClassifier(**cfg_clf))
        is_parallel = True

    if isinstance(validation, (float, int, long)) and validation > 1:
        if is_parallel:
            parallel_profile = None
        else:
            parallel_profile = 'threads-' + str(min(globals_.free_cpus(), validation))

        clf = FoldingClassifier(clf, n_folds=int(validation), parallel_profile=parallel_profile)
        lds_test = LabeledDataStorage(data=data, target=label, sample_weight=weights)  # folding-> same data for train and test

    elif isinstance(validation, data_storage.HEPDataStorage):
        lds_test = validation.get_LabeledDataStorage(columns=features)
    elif validation is None:
        make_plots = False
    elif isinstance(validation, list) and len(validation) in (1, 2):
        data_val, target_val, weights_val = _make_data(validation[0], validation[1],
                                                       conv_ori_weights=conv_vali_weights,
                                                       conv_tar_weights=conv_vali_weights)
        lds_test = LabeledDataStorage(data=data_val, target=target_val, sample_weight=weights_val)
    else:
        raise ValueError("Validation method " + validation + " not a valid choice")

    # train the classifier
    if original_data is not None:
        clf.fit(data, label, weights)  # if error "1 not in list" or similar occurs: no valid targets (None?)

    # test the classifier
    if validation is not None:
        report = ClassificationReport({clf_name: clf}, lds_test)
        n_classes = len(set(lds_test.get_targets()))
        if n_classes == 2:
            clf_score = round(report.compute_metric(metrics.RocAuc()).values()[0], 4)
            out.add_output(["ROC AUC of ", clf_name, ": ", clf_score],
                           obj_separator="", subtitle="Report of classify")
            plot_name = clf_name + ", AUC = " + str(clf_score)
            binary_test = True
        elif n_classes == 1:
            # score returns accuracy; if only one label present, it is the same as recall
            y_true = lds_test.get_targets()
            y_pred = clf.predict(lds_test.get_data())
            w_test = lds_test.get_weights()
            clf_score = clf.score(lds_test.get_data(), y_true, w_test)
            clf_score2 = accuracy_score(y_true=y_true, y_pred=y_pred)#, sample_weight=w_test)
            class_rep = classification_report(y_true, y_pred, sample_weight=w_test)
            out.add_output(class_rep, section="Classification report " + clf_name)
            out.add_output(["accuracy with sklearn (NO WEIGHTS!): ", clf_name, ", ", curve_name, ": ", clf_score2],
                           obj_separator="", subtitle="Report of classify")
            out.add_output(["recall of ", clf_name, ", ", curve_name, ": ", clf_score],
                           obj_separator="", subtitle="Report of classify")
            binary_test = False
            plot_name = clf_name + ", recall = " + str(clf_score)
        else:
            raise ValueError("Multi-label classification not supported")

    #plots
    if make_plots:

        if curve_name is not None:
            plot_name = curve_name + " " + plot_name
        report.prediction[plot_name] = report.prediction.pop(clf_name)
        report.estimators[plot_name] = report.estimators.pop(clf_name)

        if binary_test:
            out.save_fig(plt.figure(plot_title + ", ROC " + plot_name), save_fig_cfg)
            report.roc(physical_notion=True).plot(title="ROC curve of" + clf_name + " on data:" +
                                                   data_name + "\nROC AUC = " + str(clf_score))
            plt.plot([0, 1], [1, 0], 'k--')  # the fifty-fifty line

            out.save_fig(plt.figure("Learning curve" + plot_name), save_fig_cfg)
            report.learning_curve(metrics.RocAuc(), steps=1).plot(title="Learning curve of " + plot_name)

    return clf, clf_score if clf_score is not None else clf


def reweight_mc_real(reweight_data_mc, reweight_data_real, columns=None,
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
    reweight_data_real : :class:`HEPDataStorage`
        Same as *reweight_data_mc* but for the real data.
    columns : list of strings
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
    if data_tools.is_pickle(reweighter):
        return data_tools.adv_return(reweighter, save_name=reweight_saveas)

    try:
        reweighter = __REWEIGHT_MODE.get(reweighter.lower())
    except KeyError:
        raise ValueError("Reweighter invalid: " + reweighter)
    reweighter += 'Reweighter'

    # logging and writing output
    msg = ["Reweighter:", reweighter, "with config:", meta_cfg]
    logger.info(msg)
    # TODO: columns = reweight_data_mc.columns if columns is None else columns
    out.add_output(msg + ["\nData used:\n", reweight_data_mc.get_name(), " and ",
                   reweight_data_real.get_name(), "\ncolumns used for the reweighter training:\n",
                    columns], section="Training the reweighter", obj_separator=" ")

    # do the reweighting
    reweighter = getattr(hep_ml.reweight, reweighter)(**meta_cfg)
    reweighter.fit(original=reweight_data_mc.pandasDF(columns=columns),
                   target=reweight_data_real.pandasDF(columns=columns),
                   original_weight=reweight_data_mc.get_weights(),
                   target_weight=reweight_data_real.get_weights())
    return data_tools.adv_return(reweighter, save_name=reweight_saveas)


def reweight_weights(reweight_data, reweighter_trained, columns=None,
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
    new_weights = reweighter_trained.predict_weights(reweight_data.pandasDF(columns=columns),
                                        original_weight=reweight_data.get_weights())

    # write to output
    out.add_output(["Using the reweighter:\n", reweighter_trained, "\nto reweight ",
                    reweight_data.get_name()], obj_separator="")

    if normalize:
        for i in range(3):  # enhance precision
            new_weights *= new_weights.size/new_weights.sum()
    if add_weights_to_data:
        reweight_data.set_weights(new_weights)
    return new_weights


def data_ROC(original_data, target_data, features=None, classifier=None, meta_clf=True,
             curve_name=None, n_folds=3, weight_original=None, weight_target=None,
             conv_ori_weights=False, conv_tar_weights=False,
             config_clf=None, take_target_from_data=False, cfg=cfg, **kwargs):
    """ Return the ROC AUC; useful to find out, how well two datasets can be
    distinguished.

    Learn to distinguish between monte-carl data (original) and real data
    (target) and report (plot) the ROC and the AUC.

    Parameters
    ----------
    original_data : instance of :class:`HEPDataStorage`
        The original (*or* monte-carlo data)
    target_data : instance of :class:`HEPDataStorage`
        The target (*or* real data)
    features : str or list(str, str, str, ...)
        The features of the data to be used for the training.
    classifier : str or list(str, str, str, ...)
        The classifiers to be trained on the data. The following are valid
        choices:

        - 'xgb' : XGboost classifier
        - 'tmva' : the TMVA classifier
        - 'gb': Gradient Boosting classifier from scikit-learn
        - 'rdf': Random Forest classifier
        - 'ada_dt': AdaBoost over decision trees

    meta_clf : boolean
        If True, a meta-classifier will be used to "average" the results of the
        other classifiers
    curve_name : str
        Name to label the plottet ROC curve.
    n_folds : int
        Specify how many folds and checks should be made for the training/test.
        If it is 1, a normal train-test-split with 2/3 - 1/3 ratio is done.
    weight_original : numpy array 1-D [n_samples]
        The weights for the original data. Only use if you don't want to use
        the weights contained in the original_data.
    weight_target : numpy array 1-D [n_samples]
        The weights for the target data. Only use if you don't want to use
        the weights contained in the target_data.
    conv_ori_weights : boolean
        If True, *convert* the original weights to more events.
    conv_tar_weights : boolean
        If True, *convert* the target weights to more events.
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
    __IMPLEMENTED_CLF = ['xgb', 'tmva', 'gb', 'rdf', 'ada_dt']#, 'knn']

#==============================================================================
# Initialize data and classifier
#==============================================================================

    # initialize variables and setting defaults
    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    save_ext_fig_cfg = dict(meta_config.DEFAULT_EXT_SAVE_FIG, **cfg.save_ext_fig_cfg)

    curve_name = 'data' if curve_name is None else curve_name
    if classifier is None:
        classifier = 'xgb'
    elif classifier == 'all':
        classifier = __IMPLEMENTED_CLF
        classifier.remove('tmva')
    elif classifier == 'all_with_tmva':
        classifier = __IMPLEMENTED_CLF
    classifier = data_tools.to_list(classifier)
    for clf in classifier:
        assert clf in __IMPLEMENTED_CLF, str(clf) + " not a valid classifier choice"
    n_cpu = globals_.free_cpus()
    data_name = curve_name + ", " + original_data.get_name() + " and " + target_data.get_name()

    data, label, weights = _make_data(original_data, target_data, features=features,
                                     weight_target=weight_target,
                                     weight_original=weight_original,
                                     target_from_data=take_target_from_data,
                                     conv_ori_weights=conv_ori_weights,
                                     conv_tar_weights=conv_tar_weights)

    # initialize classifiers and put them together into the factory
    factory = ClassifiersFactory()
    out.add_output(["Running data_ROC with the classifiers", classifier],
                   subtitle="Separate two datasets: data_ROC")
    if 'xgb' in classifier:
        name = "XGBoost classifier"
        cfg_clf = dict(meta_config.DEFAULT_CLF_XGB, **kwargs.get('cfg_xgb', {}))
        cfg_clf.update(dict(nthreads=n_cpu, random_state=globals_.randint+4))  # overwrite entries
        clf = XGBoostClassifier(**cfg_clf)
        factory.add_classifier(name, clf)

    if 'tmva' in classifier:
        name = "TMVA classifier"
        cfg_clf = dict(meta_config.DEFAULT_CLF_TMVA, **kwargs.get('cfg_tmva', {}))
        clf = TMVAClassifier(**cfg_clf)
        factory.add_classifier(name, clf)

    if 'gb' in classifier:
        name = "GradientBoosting classifier"
        cfg_clf = dict(meta_config.DEFAULT_CLF_GB, **kwargs.get('cfg_gb', {}))
        clf = SklearnClassifier(GradientBoostingClassifier(**cfg_clf))
        factory.add_classifier(name, clf)

    if 'rdf' in classifier:
        name = "RandomForest classifier"
        cfg_clf = dict(meta_config.DEFAULT_CLF_RDF, **kwargs.get('cfg_rdf', {}))
        cfg_clf.update(dict(n_jobs=n_cpu, random_state=globals_.randint+432))
        clf = SklearnClassifier(RandomForestClassifier(**cfg_clf))
        factory.add_classifier(name, clf)

    if 'ada_dt' in classifier:
        name = "AdABoost over DecisionTree classifier"
        cfg_clf = dict(meta_config.DEFAULT_CLF_ADA, **kwargs.get('cfg_ada_dt', {}))
        cfg_clf.update(dict(random_state=globals_.randint+43))
        clf = SklearnClassifier(AdaBoostClassifier(DecisionTreeClassifier(
                                random_state=globals_.randint + 29), **cfg_clf))
        factory.add_classifier(name, clf)
    if 'knn' in classifier:
        name = "KNearest Neighbour classifier"
        cfg_clf = dict(meta_config.DEFAULT_CLF_KNN, **kwargs.get('cfg_knn', {}))
        cfg_clf.update(dict(random_state=globals_.randint+919, n_jobs=n_cpu))
        clf = SklearnClassifier(KNeighborsClassifier(**cfg_clf))
        factory.add_classifier(name, clf)

    out.add_output("The following classifiers were used to distinguish the data",
                   subtitle="List of classifiers for data_ROC", do_print=False)
    for key, val in factory.iteritems():
        out.add_output([val], section=key, do_print=False)



    if n_folds <= 1:
        train_size = 0.66 if n_folds == 1 else n_folds
        X_train, X_test, y_train, y_test, weight_train, weight_test = (
            train_test_split(data, label, weights, train_size=train_size,
                             random_state=globals_.randint+1214))
    else:
        parallel_profile = 'threads-' + str(min(n_folds, n_cpu))
        for key, val in factory.iteritems():
            factory[key] = FoldingClassifier(val, n_folds=n_folds, parallel_profile=parallel_profile)
        X_train = X_test = data  # folding scorer takes care of CV
        y_train = y_test = label
        weight_train = weight_test = weights


#==============================================================================
#     Train the classifiers
#==============================================================================

    parallel_profile = 'threads-' + str(min(len(factory.keys()), n_cpu))

    factory.fit(X_train, y_train, weight_train, parallel_profile=parallel_profile)
    report = factory.test_on(X_test, y_test, weight_test)

    # add a voting meta-classifier
    if len(factory.keys()) > 1 and meta_clf and n_folds > 1:
        # TODO: old: estimators = copy.deepcopy([(key, val) for key, val in factory.iteritems()])
        assert y_train is y_test, "train and test not the same (problem because we use folding-clf)"
        parallel_profile = 'threads-' + str(min(n_folds, n_cpu))
        meta_clf = SklearnClassifier(LogisticRegression(penalty='l2', solver='sag', n_jobs=n_cpu))
        #meta_clf = XGBoostClassifier(n_estimators=300, eta=0.1)
        # TODO: old: meta_clf = SklearnClassifier(VotingClassifier(estimators, voting='soft'))
        meta_clf = FoldingClassifier(meta_clf, n_folds=n_folds)#, parallel_profile=parallel_profile) instead meta clf parallel
        X_meta = pd.DataFrame()
        for key, val in factory.predict_proba(X_test).iteritems():
            X_meta[key] = val[:,0]
        # TODO: old: X_meta = X_train
        meta_clf.fit(X_meta, y_train)

#==============================================================================
#   Report the classification
#==============================================================================

        # voting report
        meta_report = meta_clf.test_on(X_meta, y_train)
        meta_auc = round(meta_report.compute_metric(metrics.RocAuc()).values()[0], 4)
        out.add_output(["ROC AUC of voting meta-classifier: ",
                        meta_auc], obj_separator="", subtitle="Report of data_ROC")

        # voting plots
        meta_name = "Voting meta-clf, AUC = " + str(meta_auc)
        meta_report.prediction[meta_name] = meta_report.prediction.pop('clf')
        meta_report.estimators[meta_name] = meta_report.estimators.pop('clf')
        out.save_fig(plt.figure("ROC Voting meta-classifier" + data_name), save_fig_cfg)
        meta_report.roc(physical_notion=False).plot(title="ROC curve of meta-classifier." +
                               data_name + "\nROC AUC = " + str(meta_auc))
        plt.plot([0, 1], [0, 1], 'k--')  # the fifty-fifty line

        out.save_fig(plt.figure("Learning curve Voting meta-classifier" + data_name), save_fig_cfg)
        meta_report.learning_curve(metrics.RocAuc(), steps=1).plot(title="Learning curve Voting meta-classifier")

    # factory report
    factory_auc = report.compute_metric(metrics.RocAuc())
    for key, val in factory_auc.iteritems():  # round the auc on four digits
        factory_auc[key] = round(val, 4)
    out.add_output(factory_auc, section="ROC AUC Classifiers")

    # factory plots
    out.save_fig(plt.figure("Learning curve of classifiers" + data_name), save_fig_cfg)
    report.learning_curve(metrics.RocAuc(), steps=1).plot(title="Learning curve of classifiers")

# TODO: trial, may remove if buggy
    for key, val in factory.items():
        clf = factory.pop(key)
        clf_auc = factory_auc.get(key)
        factory[key + ", AUC = " + str(round(clf_auc, 4))] = clf

    out.save_fig(plt.figure("ROC comparison " + data_name), **save_fig_cfg)
    # TODO: if trial works, remove?:
    # curve_name += "AUC = " + str(round(ROC_AUC, 3))
    for key, val in report.prediction.items():
        report.prediction[key + ", AUC = " + str(round(factory_auc.get(key), 4))] = report.prediction.pop(key)
    report.roc(physical_notion=False).plot(title="ROC curve for comparison of " + data_name)
    plt.plot([0, 1], [0, 1], 'k--')  # the fifty-fifty line
    # factory extended plots
    out.save_fig(plt.figure("feature importance shuffled " + data_name), **save_ext_fig_cfg)
    report.feature_importance_shuffling().plot()

    out.save_fig(plt.figure("correlation matrix " + data_name), **save_ext_fig_cfg)
    report.features_correlation_matrix_by_class().plot(title="Correlation matrix of data " + data_name)

    out.save_fig(plt.figure("features pdf " + data_name), **save_ext_fig_cfg)
    report.features_pdf().plot(title="Features pdf of data: " + data_name)

    out.save_fig(plt.figure("Data_ROC prediction pdf" + data_name), **save_ext_fig_cfg)
    report.prediction_pdf().plot(title="prediction pdf of different classifiers, data: " + data_name)

    # make output
    if len(factory.keys()) == 1:
        out_auc = factory_auc.values()[0]
    else:
        out_auc = None

    return out_auc


