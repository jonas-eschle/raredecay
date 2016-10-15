# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: Jonas Eschle "Mayou36"

The Machine Learning Analysis module consists of machine-learning functions
which are mostly wrappers around already existing algorithms. The expected
format of the data is a *HEPDataStorage*.

The functions serve as basic tools, which do already a lot of the work.
"""
from __future__ import division, absolute_import

import warnings
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
#from rep.metaml import ClassifiersFactory
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
from sklearn.base import BaseEstimator

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


def make_clf(clf, n_cpu=None, dict_only=False):
    """Return a classifier-dict. Takes a str, config-dict or clf-dict or clf

    Parameters
    ----------
    clf : dict or str or sklearn/REP-classifier
        There are several ways to pass the classifier to the function.

        - Pure classifier: You can pass a classifier to the method,
          either a scikit-learn or a REP classifier.
        - Classifier with name: you can name your classifier by either:

          - using a dict with {'my_clf_1': clf}
          - using a dict with {'name': 'my_clf_1', 'clf': clf}
            where clf referes to the classifier and 'my_clf_1' can be any name.
        - Configuration for a clf: Instead of instantiating the clf outside,
          you can also pass a configuration-dictionary. This has to look like:

          - {'clf_type': config-dict, 'name': 'my_clf_1'} (name is optional)
            whereas 'clf_type' has to be any of the implemented clf-types like
            'xgb', 'rdf', 'ada' etc.
        - Get a standard-clf: providing a string refering to an implemented
          clf-type, you will get a classifier using the configuration in
          :py:mod:`~raredecay.meta_config`

    n_cpu : int > -1 or None
        The number of cpus to use for this classifier. If the classifier is not
        parallelizable, an according *parallel_profile* (also see in REP-docs)
        will be created; 'threads-n' with n the number of cpus specified before.
    dict_only : boolean
        If True, only a dictionary will be returned containing the name, config,
        clf_type and parallel_profile, n_cpu.


    Returns
    -------
    out : dict
        A dictionary containing the name ('name') of the classifier as well
        as the classifier itself ('clf'). If *dict_only* is True, no clf will
        be returned but a 'clf_type' as well as a 'config' key.
        Additionally, there are more values that can be contained: if a
        configuration and not an already instantiated clf is given:

        - **parallel_profile**: the parallel-profile (for different REP-functions)
          which is set according to the n_cpus entered as well as the n_cpus
          used. If n cpus should be used, the classifier takes, the profile
          will be None. If the classifier is using only 1 cpu, the profile will
          be 'threads-n' with n = n_cpus.
        - **n_cpus**: The number of cpus used in the classifier.
    """
    __IMPLEMENTED_CLFS = ['xgb', 'gb', 'rdf', 'nn', 'ada', 'tmva', 'knn']
    output = {}
    serial_clf = False
    clf = copy.deepcopy(clf)  # make sure not to change the argument given

    # test if input is classifier, create dict
    if isinstance(clf, (BaseEstimator, Classifier)):
        clf = {'clf': clf, 'name': clf}

    # if clf is a string only, create dict with only the type specified
    if isinstance(clf, str):
        assert clf in __IMPLEMENTED_CLFS, "clf not implemented (yet. Make an issue;) )"
        clf = {'clf_type': clf, 'name': clf}

    assert isinstance(clf, dict), "Wrong data format of classifier..."

    if isinstance(clf.get('n_cpu'), int) and n_cpu is None:
        n_cpu = clf['n_cpu']
    n_cpu = 1 if n_cpu is None else n_cpu
    if n_cpu == -1:
        n_cpu = globals_.free_cpus()

    # if input is dict containing a clf, make sure it's a Sklearn one
    if len(clf) == 1 and isinstance(clf.values()[0], (BaseEstimator, Classifier)):
        key, value = clf.popitem()
        clf['name'] = key
        clf['clf'] = value
    if isinstance(clf.get('clf'), (BaseEstimator, Classifier)):
        classifier = clf['clf']
        if not isinstance(classifier, Classifier):
            classifier = SklearnClassifier(clf=classifier)
        output['clf'] = classifier
        output['name'] = clf.get('name', 'clf')
    else:
        if not clf.has_key('clf_type'):
            for imp_clf in __IMPLEMENTED_CLFS:
                if clf.has_key(imp_clf):
                    clf['clf_type'] = imp_clf
                    clf['config'] = clf[imp_clf]
        if not clf.has_key('clf_type'):
            raise ValueError("Invalid classifier, not implemented")
        if not clf.has_key('name'):
            clf['name'] = clf['clf_type']
        default_clf = dict(
            clf_type=clf['clf_type'],
            name=meta_config.DEFAULT_CLF_NAME[clf['clf_type']],
            config=meta_config.DEFAULT_CLF_CONFIG[clf['clf_type']],
        )

        clf = dict(default_clf, **clf)

        if clf['clf_type'] == 'xgb':
            # update config dict with parallel-variables and random state
            clf['config'].update(dict(nthreads=n_cpu, random_state=globals_.randint+4))  # overwrite entries
            clf_tmp = XGBoostClassifier(**clf.get('config'))
        if clf['clf_type'] == 'tmva':
            serial_clf = True
            clf_tmp = TMVAClassifier(**clf.get('config'))
        if clf['clf_type'] == 'gb':
            serial_clf = True
            clf_tmp = SklearnClassifier(GradientBoostingClassifier(**clf.get('config')))
        if clf['clf_type'] == 'rdf':
            clf['config'].update(dict(n_jobs=n_cpu, random_state=globals_.randint+432))
            clf_tmp = SklearnClassifier(RandomForestClassifier(**clf.get('config')))
        if clf['clf_type'] == 'ada':
            serial_clf = True
            clf['config'].update(dict(random_state=globals_.randint+43))
            clf_tmp = SklearnClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
                                    random_state=globals_.randint + 29), **clf.get('config')))
        if clf['clf_type'] == 'knn':
            clf['config'].update(dict(random_state=globals_.randint+919, n_jobs=n_cpu))
            clf_tmp = SklearnClassifier(KNeighborsClassifier(**clf.get('config')))
        if clf['clf_type'] == 'rdf':
            clf['config'].update(dict(n_jobs=n_cpu, random_state=globals_.randint+432))
            clf_tmp = SklearnClassifier(RandomForestClassifier(**clf.get('config')))
        if clf['clf_type'] == 'nn':
            serial_clf = meta_config.use_gpu
            clf['config'].update(dict(random_state=globals_.randint+43))
            clf_tmp = TheanetsClassifier(**clf.get('config'))

        # assign classifier to output dict
        output['clf'] = clf_tmp
        output['name'] = clf['name']
        if dict_only:
            output.pop('clf')
            del clf_tmp
            output['clf_type'] = clf['clf_type']
            output['config'] = clf['config']
        # add parallel information
        if serial_clf and n_cpu > 1:
            output['n_cpu'] = 1
            output['parallel_profile'] = 'threads-' + str(n_cpu)
        else:
            output['n_cpu'] = n_cpu
            output['parallel_profile'] = None

    return output


def backward_feature_elimination(original_data, target_data, clf, n_folds=10,
                                 features=None, max_feature_elimination=None,
                                 max_difference_to_best=0.08, direction='backward',
                                 take_target_from_data=False):
    """Train and score on each feature subset, eliminating features backwards.

    To know, which features make a big impact on the training of the clf and
    which don't, there are several techniques to find out. The most reliable,
    but also cost-intensive one, seems to be the backward feature elimination.
    A classifier gets trained first on all the features and is validated with
    the KFold-technique and the ROC AUC. Then, a feature is removed and the
    classifier is trained and tested again. This is done for all features once.
    The one where the auc drops the least is then removed and the next round
    starts from the beginning but with one feature less.

    The function ends either if:

    - no features are left
    - max_feature_elimination features have been eliminated
    - the difference between the most useless features auc and the best
      (the run done with all features in the beginning) is higher then
      max_difference_to_best

    Parameters
    ----------
    original_data : HEPDataStorage
        The original data
    target_data : HEPDataStorage
        The target data
    clf : str {'xgb, 'rdf, 'erf', 'gb', 'ada', 'nn'} or config-dict
        For possible options, see also :py:func:`~raredecay.ml_analysis.make_clf()`
    n_folds : int > 1
        How many folds you want to split your data in when doing KFold-splits
        to measure the performance of the classifier.
    features : list(str, str, str,...)
        List of strings containing the features/columns to be used for the
        hyper-optimization or feature selection.
    max_feature_elimination : int >= 1
        How many features should be eliminated before it surely stopps
    max_difference_to_best : float
        The maximum difference between the "worst" features auc and the best
        (with all features) auc before it stopps.
    take_target_from_data : boolean
        Old, will be removed. Use if target-data == None.

    Returns
    -------
    out : dict
        Return a dictionary containing the evaluation:
        - **'roc_auc'** : an ordered-dict with the feature that was removed and
          the roc auc evaluated without that feature.
        - **'scores'** : All the roc auc with every feature removed once.
          Basically a pandas DataFrame containing all results.
    """
    # initialize variables and setting defaults
    output = {}
    start_time = -1  # means: no time measurement on the way
    available_time = 1

    # start timer if time-limit is given
    if isinstance(max_feature_elimination, str):
        max_feature_elimination = max_feature_elimination.split(":")
        assert len(max_feature_elimination) == 2, "Wrong time-format. Has to be 'hhh...hhh:mm' "
        available_time = 3600 * int(max_feature_elimination[0]) + 60 * int(max_feature_elimination[1])
        start_time = timeit.default_timer()
        assert start_time > 0, "Error, start_time is <= 0, will cause error later"

    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    if features is None:
        features = original_data.columns
        meta_config.warning_occured()
        logger.warning("Feature not specified as argument to optimize_hyper_parameters." +
                       "Features for feature-optimization will be taken from data.")
    # We do not need to create more data than we well test on
    features = data_tools.to_list(features)
    assert features != [], "No features for optimization found"

    #TODO: insert time estimation for feature optimization

    # initialize data
    data, label, weights = _make_data(original_data, target_data, features=features,
                                      target_from_data=take_target_from_data)

    # initialize clf and parallel_profile
    clf_dict = make_clf(clf=clf, n_cpu=meta_config.n_cpu_max)
    clf = clf_dict['clf']
    clf_name = clf_dict['name']
    parallel_profile = clf_dict['parallel_profile']

#==============================================================================
# start backward feature elimination
#==============================================================================
    selected_features = copy.deepcopy(features)  # explicit is better than implicit
    assert len(selected_features) > 1, "Need more then one feature to perform feature selection"

    # starting feature selection
    out.add_output(["Performing feature selection with the classifier", clf_name,
                    "of the features", features], title="Feature selection: Backward elimination")
    original_clf = FoldingClassifier(clf, n_folds=n_folds,
                                     stratified=meta_config.use_stratified_folding,
                                     parallel_profile=parallel_profile)

    # "loop-initialization", get score for all features
    roc_auc = OrderedDict({})
    collected_scores = {feature: [] for feature in selected_features}
    if direction == 'backward':
        clf = copy.deepcopy(original_clf)  # required, the features attribute can not be changed somehow
        clf.fit(data[selected_features], label, weights)
        report = clf.test_on(data[selected_features], label, weights)
        max_auc = report.compute_metric(metrics.RocAuc()).values()[0]
        roc_auc = OrderedDict({'all features': round(max_auc, 4)})
        out.save_fig(figure="feature importance " + str(clf_name), importance=2, **save_fig_cfg)
        report.feature_importance_shuffling().plot()
        out.save_fig(figure="feature correlation " + str(clf_name), importance=2, **save_fig_cfg)
        report.features_correlation_matrix().plot()
        out.save_fig(figure="ROC curve " + str(clf_name), importance=2, **save_fig_cfg)
        report.roc(physics_notion=True).plot()
        out.save_fig(figure="Learning curve " + str(clf_name), importance=3, **save_fig_cfg)
        report.learning_curve(metrics.RocAuc(), steps=2, metric_label="ROC AUC").plot()

        collected_scores['features_tot'] = []

    if max_feature_elimination in (None, -1):
        n_to_eliminate = len(selected_features) - 1  # eliminate all except one
    else:
        n_to_eliminate = min([len(selected_features) - 1, max_feature_elimination])

    # do-while python-style (with if-break inside)
    while n_to_eliminate > 0:

        # initialize variable
        difference = 1  # a surely big initialisation
        n_to_eliminate -= 1
        n_features_left = len(selected_features)
        collected_scores['features_tot'].append(n_features_left)

        # iterate through the features and remove the ith each time
        for i, feature in enumerate(selected_features):
            clf = copy.deepcopy(original_clf)  # otherwise feature attribute trouble
            temp_features = copy.deepcopy(selected_features)
            del temp_features[i]  # remove ith feature for testing
            clf.fit(data[temp_features], label, weights)
            report = clf.test_on(data[temp_features], label, weights)
            temp_auc = report.compute_metric(metrics.RocAuc()).values()[0]
            collected_scores[feature].append(round(temp_auc, 4))
            # set time condition
            if available_time < timeit.default_timer() - start_time and start_time > 0:
                n_to_eliminate = 0
                break
            if max_auc - temp_auc < difference:
                difference = max_auc - temp_auc
                temp_dict = {feature: round(temp_auc, 4)}

        if difference >= max_difference_to_best:
            break
        else:
            roc_auc.update(temp_dict)
            selected_features.remove(temp_dict.keys()[0])
            max_auc = temp_dict.values()[0]
            # set time condition
            if available_time < timeit.default_timer() - start_time and start_time > 0:
                n_to_eliminate = 0

    output['roc_auc'] = roc_auc


    for key, value in collected_scores.items():
        missing_values = len(collected_scores['features_tot']) - len(value)
        if missing_values > 0:
            collected_scores[key].extend([None] * missing_values)
    temp_val = collected_scores.pop('features_tot')
    collected_scores = {'auc w/o ' + key: val for key, val in collected_scores.items()}
    collected_scores['features_tot'] = temp_val
    collected_scores = pd.DataFrame(collected_scores)
    out.add_output(["The collected scores: ", collected_scores], importance=3)
    output['scores'] = collected_scores


    if len(selected_features) > 1 and difference >= max_difference_to_best:
        out.add_output(["Removed features and roc auc: ", roc_auc,
                        "\nStopped because difference in roc auc to best was ",
                        "higher then max_difference_to_best",
                        "\nNext feature would have been: ", temp_dict],
                        subtitle="Feature selection results")
    elif len(selected_features) > 1:
        out.add_output(["Removed features and roc auc: ", roc_auc,
                        "\nFeature elimination stopped because",
                        "max_feature_elimination was reached (feature or time limit)."],
                        subtitle="Feature selection results")
    # if all features were removed
    else:
        out.add_output(["Removed features and roc auc: ", roc_auc,
                        "All features removed, loop stopped removing because no feature was left"],
                        subtitle="Feature selection results")





def optimize_hyper_parameters(original_data, target_data, clf, n_eval, features=None,
                              n_checks=10, n_folds=10, generator_type='subgrid',
                              take_target_from_data=False, **kwargs):
    """Optimize the hyperparameters of a classifier


    Parameters
    ----------
    original_data : HEPDataStorage
        The original data
    target_data : HEPDataStorage
        The target data
    clf : config-dict
        For possible options, see also :py:func:`~raredecay.ml_analysis.make_clf()`.
        The difference is, for the feature you want to have optimised, use an
        iterable instead of a single value, e.g. 'n_estimators': [1, 2, 3, 4] etc.
    n_eval : int > 1 or str "hh...hh:mm"
        How many evaluations should be done; how many points in the
        hyperparameter-space should be tested. This can either be an integer,
        which then represents the number of evaluations done or a string in the
        format of "hours:minutes" (e.g. "3:25", "1569:01" (quite long...),
        "0:12"), which represents the approximat time it should take for the
        hyperparameter-search (**not** the exact upper limit)
    features : list(str, str, str,...)
        List of strings containing the features/columns to be used for the
        hyper-optimization.
    n_checks : int >= 1
        Number of checks on *each* KFolded dataset will be done. For example,
        you split your data into 10 folds, but may only want to train/test on
        3 different ones.
    n_folds : int > 1
        How many folds you want to split your data in when doing train/test
        sets to measure the performance of the classifier.
    generator_type : str {'subgrid', 'regression', 'random'}
        The generator searches the hyper-parameter space. Different generators
        can be used using different strategies to search for the global maximum.

        - **subgrid** : For larger grids, first performe search on smaller
          subgrids to better know the rough topology of the space.
        - **regression** : using an estimator doing regression on the already
          known hyper-parameter space points to estimate where to test for
          the next one.
        - **random** : Randomly choose points in the hyper-parameter space.
    take_target_from_data : Boolean
        OUTDATED; not encouraged to use
        If True, the target-labeling (the y) will be taken from the data
        directly and not created. Otherwise, 0 will be assumed for the
        original_data and 1 for the target_data.
    """
    # initialize variables and setting defaults
    output = {}
    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    clf_dict = make_clf(clf, n_cpu=meta_config.n_cpu_max, dict_only=True)
    config_clf = clf_dict['config']
    config_clf_cp = copy.deepcopy(config_clf)

    # Create parameter for clf and hyper-search
    if features is None:
        features = original_data.columns

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
    logger.info("Maximum possible evaluations: " + str(max_eval))

    # get a time estimation and extrapolate to get n_eval
    if isinstance(n_eval, str) and (meta_config.n_cpu_max * 2 < max_eval):
        n_eval = n_eval.split(":")
        assert len(n_eval) == 2, "Wrong time-format. Has to be 'hhh...hhh:mm' "
        available_time = 3600 * int(n_eval[0]) + 60 * int(n_eval[1])

        start_timer_test = timeit.default_timer()
        elapsed_time = 1
        min_elapsed_time = 15 + 0.005 * available_time  # to get an approximate extrapolation
        n_eval_tmp = meta_config.n_cpu_max
        n_checks_tmp = 1  #time will be multiplied by actual n_checks

        #call hyper_optimization with parameters for "one" run and measure time
        out.add_output(data_out="", subtitle="Starting small test-run for time estimation.",
                       importance=2)
        # do-while loop
        clf_tmp = copy.deepcopy(clf)
        clf_tmp['config'] = config_clf_cp
        while True:
            start_timer = timeit.default_timer()
            optimize_hyper_parameters(original_data, target_data, clf=clf_tmp,
                                      n_eval=n_eval_tmp, n_folds=n_folds, n_checks=n_checks_tmp,
                                      features=features, generator_type=generator_type,
                                      take_target_from_data=take_target_from_data, time_test=True)
            elapsed_time = timeit.default_timer() - start_timer
            if elapsed_time > min_elapsed_time:
                break
            elif n_checks_tmp < n_checks:  # for small datasets, increase n_checks for testing
                n_checks_tmp = min(n_checks, np.ceil(min_elapsed_time / elapsed_time))
            else:
                n_eval_tmp *= np.ceil(min_elapsed_time / elapsed_time)  # if time to small, increase n_rounds

        elapsed_time *= np.ceil(float(n_checks) / n_checks_tmp)  # time for "one round"
        test_time = timeit.default_timer() - start_timer_test
        n_eval = (int((available_time * 0.98 - test_time) / elapsed_time)) * int(round(n_eval_tmp))  # we did just one
        if n_eval < 1:
            n_eval = meta_config.n_cpu_max
        out.add_output(["Time for one round:", round(elapsed_time, 1), "sec.",
                        " Number of evaluations:", n_eval])

    elif isinstance(n_eval, str):
        n_eval = max_eval

    n_eval = min(n_eval, max_eval)

    # We do not need to create more data than we well test on
    features = data_tools.to_list(features)

    assert grid_param != {}, "No values for optimization found"

    # initialize data
    data, label, weights = _make_data(original_data, target_data, features=features,
                                      target_from_data=take_target_from_data)

    # initialize classifier
    clf_dict['config'] = config_clf
    clf_dict = make_clf(clf=clf_dict, n_cpu=meta_config.n_cpu_max)
    clf = clf_dict['clf']
    clf_name = clf_dict['name']
    parallel_profile = clf_dict['parallel_profile']

    # rederict print output (for the hyperparameter-optimizer from rep)
    if not kwargs.get('time_test', False):
        out.add_output("Starting hyper-optimization. This might take a while, no " +
                       "output will be displayed during the process", importance=3)
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

    if not kwargs.get('time_test', False):
        out.IO_to_sys(subtitle=clf_name + " hyperparameter/feature optimization",
                      importance=4, to_end=True)
    else:
        out.IO_to_sys(importance=0)


def classify(original_data=None, target_data=None, features=None, validation=10,
             clf='xgb', plot_importance=3, extended_report=False, plot_title=None,
             curve_name=None, target_from_data=False, conv_ori_weights=False,
             conv_tar_weights=False, conv_vali_weights=False, weights_ratio=0,
             get_predictions=False):
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
    plot_importance : int {0, 1, 2, 3, 4, 5}
        The higher the importance, the more likely the plots will be showed.
        All plots should be saved anyway.
    extended_report : boolean
        If True, make extended reports on the classifier as well as on the data,
        including feature correlation, feature importance etc.
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
    predictions = {}
    make_plot = True  # used if no validation

    plot_title = "classify" if plot_title is None else plot_title
    if (original_data is None) and (target_data is not None):
        original_data, target_data = target_data, original_data  # switch places
    if original_data is not None:
        data, label, weights = _make_data(original_data, target_data, features=features,
                                          target_from_data=target_from_data,
                                          conv_ori_weights=conv_ori_weights,
                                          conv_tar_weights=conv_tar_weights,
                                          weights_ratio=weights_ratio)
        data_name = original_data.name
        if target_data is not None:
            data_name += " and " + target_data.name

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

        clf = FoldingClassifier(clf, n_folds=int(validation), parallel_profile=parallel_profile,
                                stratified=meta_config.use_stratified_folding)
        lds_test = LabeledDataStorage(data=data, target=label, sample_weight=weights)  # folding-> same data for train and test

    elif isinstance(validation, data_storage.HEPDataStorage):
        lds_test = validation.get_LabeledDataStorage(columns=features)
    elif validation in (None, False):
        make_plot = False
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
                           obj_separator="", subtitle="Report of classify",
                           importance=4)
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
            out.add_output(["accuracy NO WEIGHTS! (just for curiosity):", clf_name,
                            ",", curve_name, ":", clf_score2],
                            subtitle="Report of classify", importance=4)
            out.add_output(["recall of", clf_name, ",", curve_name, ":", clf_score],
                           importance=4)
            binary_test = False
            plot_name = clf_name + ", recall = " + str(clf_score)
        else:
            raise ValueError("Multi-label classification not supported")

    #plots

    if make_plot:  # if no validation is given, don't make plots
        if curve_name is not None:
            plot_name = curve_name + " " + plot_name
        report.prediction[plot_name] = report.prediction.pop(clf_name)
        report.estimators[plot_name] = report.estimators.pop(clf_name)

        if binary_test:
            out.save_fig(plt.figure(plot_title + " " + plot_name),
                         importance=plot_importance, **save_fig_cfg)
            report.roc(physics_notion=True).plot(title=plot_title + "\nROC curve of" + clf_name + " on data:" +
                                                   data_name + "\nROC AUC = " + str(clf_score))
            plt.plot([0, 1], [1, 0], 'k--')  # the fifty-fifty line

            out.save_fig(plt.figure("Learning curve" + plot_name),
                         importance=plot_importance, **save_fig_cfg)
            report.learning_curve(metrics.RocAuc(), steps=1).plot(title="Learning curve of " + plot_name)
        else:
            pass
            # TODO: implement learning curve with tpr metric
#            out.save_fig(plt.figure("Learning curve" + plot_name),
#                         importance=plot_importance, **save_fig_cfg)
#            report.learning_curve(metrics., steps=1).plot(title="Learning curve of " + plot_name)
        if extended_report:
            if len(data.columns) > 1:
                out.save_fig(figure="Feature importance shuffling of " + plot_name,
                             importance=plot_importance)
                report.feature_importance_shuffling().plot(
                            title="Feature importance shuffling of " + plot_name)
                out.save_fig(figure="Feature correlation matrix of " + plot_name,
                             importance=plot_importance)
                report.features_correlation_matrix().plot()
            out.save_fig(figure="Predictiond pdf of " + plot_name, importance=plot_importance)
            report.prediction_pdf(plot_type='bar').plot()

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
    out.add_output(msg + ["\nData used:\n", reweight_data_mc.name, " and ",
                   reweight_data_real.name, "\ncolumns used for the reweighter training:\n",
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
                    reweight_data.name], obj_separator="")

    if normalize:
        for i in range(1):  # old... remove? TODO
            new_weights *= new_weights.size/new_weights.sum()
    if add_weights_to_data:
        reweight_data.set_weights(new_weights)
    return new_weights

def reweight_Kfold(reweight_data_mc, reweight_data_real, n_folds=10,
                   columns=None, reweighter='gb', meta_cfg=None, score_clf='xgb',
                   add_weights_to_data=True, mcreweighted_as_real_score=False):
    """Reweight data by "itself" for *scoring* and hyper-parameters via
    Kfolding to avoid bias.

    .. warning::
       Do NOT use for the real reweighting process! (except if you really want
       to reweight the data "by itself")


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
    output = {}
    out.add_output(["Doing reweighting_Kfold with ", n_folds, " folds"],
                   title="Reweighting Kfold", obj_separator="")
    # create variables
    assert n_folds >= 1 and isinstance(n_folds, int), "n_folds has to be >= 1, its currently" + str(n_folds)
    assert isinstance(reweight_data_mc, data_storage.HEPDataStorage), "wrong data type. Has to be HEPDataStorage, is currently" + str(type(reweight_data_mc))
    assert isinstance(reweight_data_real, data_storage.HEPDataStorage), "wrong data type. Has to be HEPDataStorage, is currently" + str(type(reweight_data_real))



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
        plot_importance1 = 4 if fold == 0 else 1
        if n_folds > 1:
            train_real.plot(figure="Reweighter trainer, example, fold " + str(fold),
                            importance=plot_importance1)
            train_mc.plot(figure="Reweighter trainer, example, fold " + str(fold),
                          importance=plot_importance1)

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
        if n_folds > 1:
            out.save_fig("new weights of fold " + str(fold))
            plt.hist(new_weights, bins=40, log=True)

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

    out.save_fig(figure="New weights of total mc", importance=4)
    plt.hist(new_weights_all, bins=30, log=True)
    plt.title("New weights of reweighting with Kfold")

    # create score
    if mcreweighted_as_real_score:
        out.add_output("", subtitle="Kfold reweight report", section="Precision scores of classification on reweighted mc")
        score_list = [("Reweighted: ", scores, 'score_reweighted'),
                      ("mc as real (min): ", score_min, 'score_min'),
                      ("real as real (max): ", score_max, 'score_max')]

        for name, score, key in score_list:
            mean, std = round(np.mean(score), 4), round(np.std(score), 4)
            out.add_output(["Classify the target, average score " + name + str(mean) +
                            " +- " + str(std)], to_end=True)
            output[key] = mean

    new_weights_all = pd.Series(new_weights_all, index=new_weights_index)

    output['weights'] = new_weights_all
    return output


if __name__ == "main":
    print 'test'
