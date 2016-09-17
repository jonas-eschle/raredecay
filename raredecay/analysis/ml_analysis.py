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
import timeit
from collections import Counter, OrderedDict

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
from rep.estimators.interface import Classifier

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
    """Return the concatenated data, weights and labels for classifier training.

     Differs to only *make_dataset* from the HEPDataStorage by providing the
     possibility of using other weights.
    """
    # make temporary weights if specific weights are given as parameters
    temp_ori_weights = None
    temp_tar_weights = None
    if not dev_tool.is_in_primitive(weight_original, None):
        temp_ori_weights = original_data.get_weights()
        original_data.set_weights(weight_original)
    if not dev_tool.is_in_primitive(weight_target, None):
        temp_tar_weights = target_data.get_weights()
        target_data.set_weights(weight_target)

    # create the data, target and weights
    data_out = original_data.make_dataset(target_data, columns=features,
                                          targets_from_data=target_from_data,
                                          weights_as_events=conv_ori_weights,
                                          weights_as_events_2=conv_tar_weights,
                                          weights_ratio=weights_ratio)

    # reassign weights if specific weights have been used
    if not dev_tool.is_in_primitive(temp_ori_weights, None):
        original_data.set_weights(temp_ori_weights)
    if not dev_tool.is_in_primitive(temp_tar_weights, None):
        original_data.set_weights(temp_tar_weights)

    return data_out


def optimize_hyper_parameters(original_data, target_data, clf, config_clf, n_eval,
                              features=None, optimize_features=False,
                              n_checks=10, n_folds=10, generator_type=None,
                              take_target_from_data=False, train_best=False):
    """Optimize the hyperparameters of a classifier or perform feature selection

    Parameters
    ----------
    original_data : HEPDataStorage
        The original data
    target_data : HEPDataStorage
        The target data
    clf : str {'xgb, 'rdf, 'erf', 'gb', 'ada', 'nn'}
        The name of the classifier
    config_clf : dict
        The configuration of the classifier
    n_eval : int > 1 or str "hh...hh:mm"
        How many evaluations should be done; how many points in the
        hyperparameter-space should be tested. This can either be an integer,
        which then represents the number of evaluations done or a string in the
        format of "hours:minutes" (e.g. "3:25", "1569:01" (quite long...),
        "0:12"), which represents the approximat time it should take for the
        hyperparameter-search (**not** the exact upper limit)
    features : list(str, str, str,...)
        List of strings containing the features/columns to be used for the
        hyper-optimization or feature selection.
    optimize_features : Boolean
        If True, feature selection will be done instead of hyperparameter-
        optimization
    n_checks : int >= 1
        Number of checks on *each* KFolded dataset will be done. For example,
        you split your data into 10 folds, but may only want to train/test on
        3 different ones.
    n_folds : int > 1
        How many folds you want to split your data in when doing train/test
        sets to measure the performance of the classifier.
    take_target_from_data : Boolean
        If True, the target-labeling (the y) will be taken from the data
        directly and not created. Otherwise, 0 will be assumed for the
        original_data and 1 for the target_data.
    train_best : boolean
        If True, train the best classifier (if hyperparameter-optimization is
        done) on the full dataset.
    """

    # initialize variables and setting defaults
    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    save_ext_fig_cfg = dict(meta_config.DEFAULT_EXT_SAVE_FIG, **cfg.save_ext_fig_cfg)
    config_clf_cp = copy.deepcopy(config_clf)

    # Create parameter for clf and hyper-search
    if not dev_tool.is_in_primitive(config_clf.get('features', None)):
        features = config_clf.get('features', None)
    if optimize_features:
        if features is None:
            features = original_data.columns
            meta_config.warning_occured()
            logger.warning("Feature not specified in classifier or as argument to optimize_hyper_parameters." +
                           "Features for feature-optimization will be taken from data.")


    else:
        grid_param = {}
        list_param = ['layers', 'trainers']  # parameters which are by their nature a list, like nn-layers
        for key, val in config_clf.items():
            if isinstance(val, (list, np.ndarray, pd.Series)):
                if key not in list_param or isinstance(val[0], list):
                    val = data_tools.to_list(val)
                    grid_param[key] = config_clf.pop(key)



        # count maximal combinations of parameters
        max_eval = 1
        for n_params in grid_param.itervalues():
            max_eval *= len(n_params)

        # get a time estimation and extrapolate to get n_eval
        if isinstance(n_eval, str) and meta_config.n_cpu_max * 2 < max_eval:
            n_eval = n_eval.split(":")
            assert len(n_eval) == 2, "Wrong time-format. Has to be 'hhh...hhh:mm' "
            available_time = 3600 * int(n_eval[0]) + 60 * int(n_eval[1])

            start_timer_test = timeit.default_timer()
            elapsed_time = 1
            min_elapsed_time = 15 + 0.005 * available_time  # to get an approximate extrapolation
            n_eval_tmp = meta_config.n_cpu_max
            n_checks_tmp = 1  #time will be multiplied by actual n_checks
            #call hyper_optimization with parameters for "one" run and measure time
            out.add_output(data_out="", subtitle="Test run for time estimation only!")
            # do-while loop
            while True:
                start_timer = timeit.default_timer()
                config_clf_cp1 = copy.deepcopy(config_clf_cp)
                optimize_hyper_parameters(original_data, target_data, clf=clf, config_clf=config_clf_cp1,
                                          n_eval=n_eval_tmp, n_folds=n_folds, n_checks=n_checks_tmp,
                                          features=features, generator_type=generator_type,
                                          optimize_features=False, train_best=False,
                                          take_target_from_data=take_target_from_data)
                elapsed_time = timeit.default_timer() - start_timer
                if elapsed_time > min_elapsed_time:
                    break
                elif n_checks_tmp < n_checks:  # for small datasets, increase n_checks for testing
                    n_checks_tmp = n_checks
                else:
                    n_eval_tmp *= np.ceil(min_elapsed_time / elapsed_time)  # if time to small, increase n_rounds

            elapsed_time *= np.ceil(float(n_checks) / n_checks_tmp)  # time for "one round"
            test_time = timeit.default_timer() - start_timer_test
            n_eval = (int((available_time * 0.98 - test_time) / elapsed_time)) * n_eval_tmp  # we did just one
            if n_eval < 1:
                n_eval = meta_config.n_cpu_max
            out.add_output(["Time for one round:", elapsed_time, "Number of evaluations:", n_eval])

        elif isinstance(n_eval, str):
            n_eval = max_eval

        n_eval = min(n_eval, max_eval)


    # We do not need to create more data than we well test on
    features = data_tools.to_list(features)
    if optimize_features:
        grid_param = features

    #TODO: insert time estimation




    assert grid_param != {}, "No values for optimization found"

    # parallelize on the lowest level if possible (uses less RAM)
    if clf in ('xgb', 'rdf', 'erf'):
        parallel_profile = None
    else:
        parallel_profile = 'threads-' + str(min(globals_.free_cpus(), n_eval))


    # initialize data
    data, label, weights = _make_data(original_data, target_data, features=features,
                                      target_from_data=take_target_from_data)

    # initialize classifier
    if clf == 'xgb':
        clf_name = "XGBoost"
        config_clf.update(nthreads=globals_.free_cpus())
        clf = XGBoostClassifier(**config_clf)
    elif clf == 'rdf':
        clf_name = "Random Forest"
        config_clf.update(n_jobs=globals_.free_cpus())  # needs less RAM if parallelized this way
        clf = SklearnClassifier(RandomForestClassifier(**config_clf))
    elif clf == 'gb':
        clf_name = "GradientBoosting classifier"
        clf = SklearnClassifier(GradientBoostingClassifier(**config_clf))
    elif clf == 'nn':
        clf_name = "Theanets Neural Network"
        parallel_profile = None if meta_config.use_gpu else parallel_profile
        clf = TheanetsClassifier(**config_clf)
    elif clf == 'erf':
        clf_name = "Extra Random Forest"
        config_clf.update(n_jobs=globals_.free_cpus())
        clf = SklearnClassifier(ExtraTreesClassifier(**config_clf))
    elif clf == 'ada':
        clf_name = "AdaBoost classifier"
        clf = SklearnClassifier(AdaBoostClassifier(**config_clf))

    if optimize_features:
        selected_features = copy.deepcopy(features)  # explicit is better than implicit
        assert len(selected_features) > 1, "Need more then one feature to perform feature selection"

        # starting feature selection
        out.add_output(["Performing feature selection of classifier", clf, "of the features", features],
                       obj_separator=" ", title="Feature selection")
        original_clf = FoldingClassifier(clf, n_folds=n_folds,
                                         parallel_profile=parallel_profile)

        # "loop-initialization", get score for all features
        clf = copy.deepcopy(original_clf)  # required, the features attribute can not be changed somehow
        clf.fit(data[selected_features], label, weights)
        report = clf.test_on(data[selected_features], label, weights)
        max_auc = report.compute_metric(metrics.RocAuc()).values()[0]
        roc_auc = OrderedDict({'all features': round(max_auc, 4)})
        out.save_fig(figure="feature importance " + str(clf_name))
        report.feature_importance_shuffling().plot()
        out.save_fig(figure="feature correlation " + str(clf_name))
        report.features_correlation_matrix().plot()

        # do-while python-style (with if-break inside)
        while len(selected_features) > 1:

            # initialize variable
            difference = 1  # a surely big initialisation

            # iterate through the features and remove the ith each time
            for i, feature in enumerate(selected_features):
                clf = copy.deepcopy(original_clf)  # otherwise feature attribute trouble
                temp_features = selected_features[:]
                del temp_features[i]  # remove ith feature for testing
                clf.fit(data[temp_features], label, weights)
                report = clf.test_on(data[temp_features], label, weights)
                temp_auc = report.compute_metric(metrics.RocAuc()).values()[0]
                if max_auc - temp_auc < difference:
                    difference = max_auc - temp_auc
                    temp_dict = {feature: round(temp_auc, 4)}

            if difference >= meta_config.max_difference_feature_selection:
                break
            else:
                roc_auc.update(temp_dict)
                selected_features.remove(temp_dict.keys()[0])
                max_auc = temp_dict.values()[0]

        if len(selected_features) > 1:
            out.add_output(["ROC AUC if the feature was removed", roc_auc,
                            "next feature", temp_dict],
                            subtitle="Feature selection results")
        # if all features were removed
        else:
            out.add_output(["ROC AUC if the feature was removed", roc_auc,
                            "All features removed, loop stopped removing because, no feature was left"],
                            subtitle="Feature selection results")

    else:
        # rederict print output (for the hyperparameter-optimizer from rep)
        out.IO_to_string()

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
                report.feature_importance_shuffling().plot(title=name4)
            except:
                logger.error("Could not plot hyper optimization " + clf_name + " plots")

        out.IO_to_sys(subtitle=clf_name + " hyperparameter/feature optimization")


def classify(original_data=None, target_data=None, features=None, validation=10, clf='xgb',
             make_plots=True, plot_title=None, curve_name=None, target_from_data=False,
             conv_ori_weights=False, conv_tar_weights=False, conv_vali_weights=False,
             weights_ratio=0, get_predictions=False):
    """Training and testing a classifier or distinguish a dataset

    Classify is a multi-purpose function which does most of the things around
    machine-learning. It can be used for:

    - Training a clf.
        A quite simple task. You give some data, specify a clf and set
        validation to False (not mandatory actually, but pay attention if
        validation is set to an integer)
    - Predict data.
        Use either a pre-trained (see above) classifier or specify one with a
        string and give in some data to the validation and no to the
        original_data or target_data. Set get_predictions to True and you're
        done.
    - Get a ROC curve of two datasets.
        Specify the two input data (original_data and target_data) and use
        cross-validation by setting validation to the number of folds

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
    clf : str {'xgb', 'rdf'} or REP-classifier
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
    get_predictions : boolean
        If True, return a dictionary containing the prediction probabilities, the
        true y-values and maybe, in the futur, even more.

    Returns
    -------
    out : clf
        Return the trained classifier.
    .. note::
        If validation was choosen to be KFold, the returned classifier well be
        instance of :py:class:`~rep.metaml.folding.FoldingClassifier()`!
    out : float (only if validation is not None)
        Return the score (recall or roc auc) of the validation. If only one
        class (sort of labels, mostly if data for validation is provided) is
        given, the recall will be computed. Otherwise the ROC-AUC (like for
        cross-validation)
    out : dict  (only if *get_predictions* is True)
        Return a dict containing the predictions, probability and more.

        - 'y_pred' : predictions of the classifier
        - 'y_proba' : prediciton probabilities
        - 'y_true' : the true labels of the data (if available)
        - 'weights' : the weights of the corresponding predicitons
    """
    logger.info("Starting classify with " + str(clf))
    # initialize variables and data
    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    save_ext_fig_cfg = dict(meta_config.DEFAULT_EXT_SAVE_FIG, **cfg.save_ext_fig_cfg)
    predictions = {}

    plot_title = "classify" if plot_title is None else plot_title
    if (original_data is None) and (target_data is not None):
        original_data, target_data = target_data, original_data  # switch places
    if original_data is not None:
        data, label, weights = _make_data(original_data, target_data, features=features,
                                          target_from_data=target_from_data,
                                          conv_ori_weights=conv_ori_weights,
                                          conv_tar_weights=conv_tar_weights,
                                          weights_ratio=weights_ratio)
        data_name = original_data.get_name()
        if target_data is not None:
            data_name += " and " + target_data.get_name()
    clf_name = 'classifier'
    if clf == 'xgb':
        clf_name = 'XGBoost classifier'
        clf = XGBoostClassifier(**dict(meta_config.DEFAULT_CLF_XGB, nthreads=globals_.free_cpus()))
        is_parallel = True
    elif clf == 'rdf':
        clf_name = "RandomForest classifier"
        cfg_clf = meta_config.DEFAULT_CLF_RDF,
        cfg_clf = dict(n_jobs=globals_.free_cpus(), random_state=globals_.randint+432)
        clf = SklearnClassifier(RandomForestClassifier(**cfg_clf))
        is_parallel = True
    elif clf == 'nn':
        clf_name = "Theanets Neural Network"
        cfg_clf = dict(meta_config.DEFAULT_CLF_NN, random_state=globals_.randint+1817)
        clf = TheanetsClassifier(**cfg_clf)
        is_parallel = True
    elif isinstance(clf, Classifier):
        is_parallel = True
    else:
        raise ValueError('Not a valid classifier or string for a clf')


    if isinstance(validation, (float, int, long)) and validation > 1:
        if is_parallel:
            parallel_profile = None
        else:
            parallel_profile = 'threads-' + str(min(globals_.free_cpus(), validation))

        clf = FoldingClassifier(clf, n_folds=int(validation), parallel_profile=parallel_profile)
        lds_test = LabeledDataStorage(data=data, target=label, sample_weight=weights)  # folding-> same data for train and test

    elif isinstance(validation, data_storage.HEPDataStorage):
        lds_test = validation.get_LabeledDataStorage(columns=features)
    elif validation in (None, False):
        make_plots = False
        clf_score = None
    elif isinstance(validation, list) and len(validation) in (1, 2):
        data_val, target_val, weights_val = _make_data(validation[0], validation[1],
                                                       conv_ori_weights=conv_vali_weights,
                                                       conv_tar_weights=conv_vali_weights)
        lds_test = LabeledDataStorage(data=data_val, target=target_val, sample_weight=weights_val)
    else:
        raise ValueError("Validation method " + str(validation) + " not a valid choice")

    # train the classifier
    if original_data is not None:
        clf.fit(data, label, weights)  # if error "1 not in list" or similar occurs: no valid targets (None?)

    # test the classifier
    if validation not in (None, False):
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
            y_pred_proba = clf.predict_proba(lds_test.get_data())
            if get_predictions:
                predictions['y_proba'] = y_pred_proba
                predictions['y_pred'] = y_pred
                predictions['y_true'] = y_true
                predictions['weights'] = lds_test.get_weights(allow_nones=True)
            w_test = lds_test.get_weights()
            clf_score = clf.score(lds_test.get_data(), y_true, w_test)
            clf_score2 = accuracy_score(y_true=y_true, y_pred=y_pred)#, sample_weight=w_test)
            class_rep = classification_report(y_true, y_pred, sample_weight=w_test)
            out.add_output(class_rep, section="Classification report " + clf_name)
            out.add_output(["accuracy NO WEIGHTS! (just for curiosity): ", clf_name, ", ", curve_name, ": ", clf_score2],
                           obj_separator="", subtitle="Report of classify")
            out.add_output(["recall of ", clf_name, ", ", curve_name, ": ", clf_score],
                           obj_separator="")
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
            report.roc(physics_notion=True).plot(title="ROC curve of" + clf_name + " on data:" +
                                                   data_name + "\nROC AUC = " + str(clf_score))
            plt.plot([0, 1], [1, 0], 'k--')  # the fifty-fifty line

            out.save_fig(plt.figure("Learning curve" + plot_name), save_fig_cfg)
            report.learning_curve(metrics.RocAuc(), steps=1).plot(title="Learning curve of " + plot_name)

    if clf_score is None:
        return clf
    elif get_predictions:
        return clf, clf_score, predictions
    else:
        return clf, clf_score

    #return clf, clf_score if clf_score is not None else clf


def reweight_train(reweight_data_mc, reweight_data_real, columns=None,
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
       But be aware, that you can easily screw up things by overfitting.

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

    if columns is None:
        # use the intesection of both colomns
        common_cols = set(reweight_data_mc.columns)
        common_cols.intersection_update(reweight_data_real.columns)
        columns = list(common_cols)
        if columns != reweight_data_mc.columns or columns != reweight_data_real.columns:
            logger.warning("No columns specified for reweighting, took intersection" +
                           " of both dataset, as it's columns are not equal." +
                           "\nTherefore some columns were not used!")
            meta_config.warning_occured()

    # train the reweighter
# TODO: remove next line, accidentialy inserted?
    # hep_ml.reweight.BinsReweighter()
    reweighter = getattr(hep_ml.reweight, reweighter)(**meta_cfg)
    reweighter.fit(original=reweight_data_mc.pandasDF(columns=columns),
                   target=reweight_data_real.pandasDF(columns=columns),
                   original_weight=reweight_data_mc.get_weights(),
                   target_weight=reweight_data_real.get_weights())
    return data_tools.adv_return(reweighter, save_name=reweight_saveas)


def reweight_weights(reweight_data, reweighter_trained, columns=None,
                     normalize=True, add_weights_to_data=True):
    """Add (or only return) new weights to the data by applying a given
    reweighter on the reweight_data.

    Can be seen as a wrapper for the
    :py:func:`~hep_ml.reweight.GBReweighter.predict_weights` method.
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
    out.add_output(["Using the reweighter:\n", reweighter_trained, "\n to reweight ",
                    reweight_data.get_name()], obj_separator="")

    if normalize:
        for i in range(1):  # enhance precision
            new_weights *= new_weights.size/new_weights.sum()
    if add_weights_to_data:
        reweight_data.set_weights(new_weights)
    return new_weights

def reweight_Kfold(reweight_data_mc, reweight_data_real, n_folds=10, make_plot=True,
                   columns=None, reweighter='gb', meta_cfg=None,
                   add_weights_to_data=True, mcreweighted_as_real_score=False):
    """Reweight data by "itself" for *scoring* and hyper-parameters via
    Kfolding to avoid bias.

    .. warning::
       Do NOT use for the real reweighting process!


    If you want to figure out the hyper-parameters for a reweighting process
    or just want to find out how good the reweighter works, you may want to
    apply this to the data itself. This means:

    - train a reweighter on mc/real
    - apply it to get new weights for mc
    - compare the mc/real distribution

    The problem arises with biasing your reweighter. As in classification
    tasks, where you split your data into train/test sets for Kfolds, you
    want to do the same here. Therefore:

    - split the mc data into (n_folds-1)/n_folds (training)
    - train the reweighter on the training mc/complete real (if
      mcreweighted_as_real_score is True, the real data will be folded too
      for unbiasing the score)
    - reweight the leftout mc test-fold
    - do this n_folds times
    - getting unbiased weights

    The parameters are more or less the same as for the
    :py:func:`~raredecay.analysis.ml_analysis.reweight_train` and
    :py:func:`~raredecay.analysis.ml_analysis.reweight_weights`

    Parameters
    ----------
    reweight_data_mc : :class:`HEPDataStorage`
        The Monte-Carlo data, which has to be "fitted" to the real data.
    reweight_data_real : :class:`HEPDataStorage`
        Same as *reweight_data_mc* but for the real data.
    n_folds : int >= 1
        The number of folds to split the data. Usually, the more folds the
        "better" reweighting.

        If n_folds = 1, the data will be reweighted directly and the benefit
        of Kfolds and the unbiasing *disappears*
    make_plot : boolean or str
        If True, an example data plot as well as the final weights will be
        plotted.

        If 'all', all the data folds will be plotted.

        If False, no plots at all will be made.

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
    add_weights_to_data : boolean
        If True, the new weights will be added (in place) to the mc data and
        returned. Otherwise, the weights will only be returned.
    mcreweighted_as_real_score : boolean or str
        If a string, it has to be an implemented classifier in *classify*.
        If true, the default ('xgb' most probably) will be used.

        If not False, calculate and print the score. This scoring is based on a
        clf, which was trained on the not reweighted mc and real data and
        tested on the reweighted mc, and then predicts how many it "thinks"
        are real datapoints.

        Intuitively, a classifiers learns to distinguish between mc and real
        and then classifies mc reweighted data labeled as real; he says, how
        "real" the reweighted data looks like. So a higher score is better.
        Drawback of this method is, it is completely blind to over-fitting
        of the reweighter. To get a relation, the classifier also predicts
        the mc (which should be an under limit) as well as the real data
        (which should be an upper limit).

        Even dough this scoring sais not a lot about how well the reweighting
        worked, we can say, that if the score is higher than the real one,
        it has somehow over-fitted (if a classifier cannot classify, say,
        more than 70% of the real data as real, it should not be able to
        classify more than 70% of the reweighted mc as real. Reweighted mc
        should not "look more real" than real data)
    out : numpy array
        Return the new weights

    """
    out.add_output(["Doing reweighting_Kfold with ", n_folds, " folds"],
                   title="Reweighting Kfold", obj_separator="")
    # create variables
    assert n_folds >= 1 and isinstance(n_folds, int), "n_folds has to be >= 1, its currently" + str(n_folds)
    assert isinstance(reweight_data_mc, data_storage.HEPDataStorage), "wrong data type. Has to be HEPDataStorage, is currently" + str(type(reweight_data_mc))
    assert isinstance(reweight_data_real, data_storage.HEPDataStorage), "wrong data type. Has to be HEPDataStorage, is currently" + str(type(reweight_data_real))
    if isinstance(mcreweighted_as_real_score, str):
        score_clf = mcreweighted_as_real_score
        mcreweighted_as_real_score = True
    elif mcreweighted_as_real_score:
        score_clf = 'xgb'

    new_weights_all = []
    new_weights_index = []

    if mcreweighted_as_real_score:
        scores = np.ones(n_folds)
        score_min = np.ones(n_folds)
        score_max = np.ones(n_folds)


    # split data to folds and loop over them
    reweight_data_mc.make_folds(n_folds=n_folds)
    reweight_data_real.make_folds(n_folds=n_folds)
    logger.info("Data created, starting folding")
    for fold in range(n_folds):

        # create train/test data
        if n_folds > 1:
            train_real, test_real = reweight_data_real.get_fold(fold)
        else:
            train_real = test_real = reweight_data_real.get_fold(fold)
        if n_folds > 1:
            train_mc, test_mc = reweight_data_mc.get_fold(fold)
        else:
            train_mc = test_mc = reweight_data_mc

        if mcreweighted_as_real_score:
            old_mc_weights = test_mc.get_weights()

        # plot the first fold as example (the first one surely exists)
        if ((fold == 0) and make_plot) or make_plot == 'all':
            train_real.plot(figure="Reweighter trainer, example, fold " + str(fold))
            train_mc.plot(figure="Reweighter trainer, example, fold " + str(fold))

        # train reweighter on training data
        reweighter_trained = reweight_train(reweight_data_mc=train_mc,
                                            reweight_data_real=train_real,
                                            columns=columns, reweighter=reweighter,
                                            meta_cfg=meta_cfg)
        logger.info("reweighting fold " + str(fold) + "finished")

        new_weights = reweight_weights(reweight_data=test_mc, columns=columns,
                                       reweighter_trained=reweighter_trained,
                                       add_weights_to_data=True)  # fold only, not full data
        # plot one for example of the new weights
        if (((fold == 0) and make_plot) or make_plot == 'all') and n_folds > 1:
            plt.figure("new weights of fold " + str(fold))
            plt.hist(new_weights,bins=40, log=True)

        if mcreweighted_as_real_score:
            # treat reweighted mc data as if it were real data target(1)
            test_mc.set_targets(1)
            # train clf on real and mc and see where it classifies the reweighted mc
            clf, scores[fold] = classify(train_mc, train_real, validation=test_mc,
                                         curve_name="mc reweighted as real",
                                         plot_title="fold " + str(fold) + " reweighted validation",
                                         weights_ratio=1, clf=score_clf)

            # Get the max and min for "calibration" of the possible score for the reweighted data by
            # passing in mc and label it as real (worst/min score) and real labeled as real (best/max)
            test_mc.set_weights(old_mc_weights)  # TODO: check, was new implemented. Before was 1
            tmp_, score_min[fold] = classify(clf=clf, validation=test_mc,
                                             curve_name="mc as real")
            test_real.set_targets(1)
            tmp_, score_max[fold] = classify(clf=clf, validation=test_real,
                                             curve_name="real as real")


        # collect all the new weights to get a really cross-validated reweighted dataset
        new_weights_all.append(new_weights)
        new_weights_index.append(test_mc.get_index())

        logger.info("fold " + str(fold) + "finished")
        # end of for-loop

    #concatenate weights and index
    if n_folds == 1:
        new_weights_all = np.array(new_weights_all)
        new_weights_index = np.array(new_weights_index)
    else:
        new_weights_all = np.concatenate(new_weights_all)
        new_weights_index = np.concatenate(new_weights_index)
    if add_weights_to_data:
        reweight_data_mc.set_weights(new_weights_all, index=new_weights_index)

    if make_plot:
        out.save_fig(figure="New weights of total mc")
        plt.hist(new_weights_all, bins=30, log=True)
        plt.title("New weights of reweighting with Kfold")

    # create score
    if mcreweighted_as_real_score:
        out.add_output("", subtitle="Kfold reweight report", section="Precision scores of classification on reweighted mc")
        score_list = [("Reweighted: ", scores), ("mc as real (min): ", score_min), ("real as real (max): ", score_max)]
        for name, score in score_list:
            mean, std = round(np.mean(score), 4), round(np.std(score), 4)
            out.add_output(["Classify the target, average score " + name + str(mean) + " +- " + str(std)])

    return new_weights_all


# TODO: continue cleaning up the code from here down
def data_ROC(original_data, target_data, features=None, classifier=None, meta_clf=True,
             curve_name=None, n_folds=3, weight_original=None, weight_target=None,
             conv_ori_weights=False, conv_tar_weights=False, weights_ratio=0,
             config_clf=None, take_target_from_data=False, cfg=cfg, **kwargs):
    """.. caution:: This method is maybe outdated and should be used with caution!

    Return the ROC AUC; useful to find out, how well two datasets can be
    distinguished.

    Learn to distinguish between monte-carl data (original) and real data
    (target) and report (plot) the ROC and the AUC.

    The original and the target data are concatenated, mixed and split up
    into n folds. A classifier gets trainer on the train data set and
    validated on the test data for n folds.

    .. note:: This method of finding how well two datasets are separabel is
    good in general but has its problems if you have only a few data or
    comparably big (or wide spread) weights.

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
                                     weight_target=weight_target, weights_ratio=weights_ratio,
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
        meta_report.roc(physics_notion=False).plot(title="ROC curve of meta-classifier." +
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
    report.roc(physics_notion=False).plot(title="ROC curve for comparison of " + data_name)
    plt.plot([0, 1], [0, 1], 'k--')  # the fifty-fifty line

    # factory extended plots
    try:
        out.save_fig(plt.figure("feature importance shuffled " + data_name), **save_ext_fig_cfg)
        report.feature_importance_shuffling().plot()
    except AttributeError:
        meta_config.error_occured()
        logger.error("could not determine feature_importance_shuffling in Data_ROC")

    try:
        out.save_fig(plt.figure("correlation matrix " + data_name), **save_ext_fig_cfg)
        report.features_correlation_matrix_by_class().plot(title="Correlation matrix of data " + data_name)
    except AttributeError:
        meta_config.error_occured()
        logger.error("could not determine features_correltaion_matrix_by_class in Data_ROC")



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


