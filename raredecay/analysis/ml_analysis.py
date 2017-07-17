# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: Jonas Eschle "Mayou36"

The Machine Learning Analysis module consists of machine-learning functions
which are mostly wrappers around already existing algorithms.

Several "new types" of formats are introduced by using the available formats
from all the libraries (scikit-learn, pandas, numpy etc) and brings together
what belongs together. It takes away all the unnecessary work done so many
times for the simple tasks.

The functions serve as basic tools, which do already a lot of the work.
"""
from __future__ import division, absolute_import


import copy
import timeit
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hep_ml imports
import hep_ml.reweight

# scikit-learn imports
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier  # , VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report  # recall_score,

# import Reproducible Experimental Platform
from rep.data import LabeledDataStorage

from rep.estimators import SklearnClassifier, XGBoostClassifier, TMVAClassifier
#from rep.estimators.theanets import TheanetsClassifier
from rep.estimators.interface import Classifier

from rep.metaml.folding import FoldingClassifier
from rep.metaml import GridOptimalSearchCV, FoldingScorer, RandomParameterOptimizer
from rep.metaml import SubgridParameterOptimizer
from rep.metaml.gridsearch import RegressionParameterOptimizer  # , AnnealingParameterOptimizer

from rep.report import metrics
from rep.report.classification import ClassificationReport

# raredecay imports
from raredecay.tools import dev_tool, data_tools, data_storage
from raredecay.globals_ import out
# from raredecay import globals_

# import configuration
import importlib
from raredecay import meta_config
cfg = importlib.import_module(meta_config.run_config)
logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


def _make_data(original_data, target_data=None, features=None, target_from_data=False,
               weights_ratio=0, weights_original=None, weights_target=None):
    """Return the concatenated data, weights and labels for classifier training.

     Differs to only *make_dataset* from the HEPDataStorage by providing the
     possibility of using other weights.
    """
    # make temporary weights if specific weights are given as parameters
    temp_ori_weights = None
    temp_tar_weights = None
    if not dev_tool.is_in_primitive(weights_original, None):
        temp_ori_weights = original_data.get_weights()
        original_data.set_weights(weights_original)
    if not dev_tool.is_in_primitive(weights_target, None):
        temp_tar_weights = target_data.get_weights()
        target_data.set_weights(weights_target)

    # create the data, target and weights
    data_out = original_data.make_dataset(target_data, columns=features,
                                          targets_from_data=target_from_data,
                                          weights_ratio=weights_ratio)

    # reassign weights if specific weights have been used
    if not dev_tool.is_in_primitive(temp_ori_weights, None):
        original_data.set_weights(temp_ori_weights)
    if not dev_tool.is_in_primitive(temp_tar_weights, None):
        original_data.set_weights(temp_tar_weights)

    return data_out


def make_clf(clf, n_cpu=None, dict_only=False):
    """Return a classifier-dict. Takes a str, config-dict or clf-dict or clf.

    This function is used to bring classifiers into the "same" format. It
    takes several kind of arguments, extracts the information, sorts it and
    creates an instance of a classifier if needed.

    Currently implemented classifiers are found below

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

          - {'*clf_type*': config-dict, 'name': 'my_clf_1'} (name is optional)
            whereas 'clf_type' has to be any of the implemented clf-types like
            'xgb', 'rdf', 'ada' etc.
        - Get a standard-clf: providing a *string* only refering to an implemented
          clf-type, you will get a classifier using the configuration in
          :py:mod:`~raredecay.meta_config`

    n_cpu : int or None
        The number of cpus to use for this classifier. If the classifier is not
        parallelizable, an according *parallel_profile* (also see in REP-docs)
        will be created; 'threads-n' with n the number of cpus specified before.

        .. warning::
            This overwrites the global n-cpu settings for this specific classifier
    dict_only : boolean
        If True, only a dictionary will be returned containing the name, config,
        clf_type and parallel_profile, n_cpu, but no classifier instance will
        be created.


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
    #: Currently implemented classifiers:
    __IMPLEMENTED_CLFS = ['xgb', 'gb', 'rdf', 'nn', 'ada', 'tmva', 'knn']
    output = {}
    serial_clf = False
    clf = copy.deepcopy(clf)  # make sure not to change the argument given

    # test if input is classifier, create dict
    if isinstance(clf, (BaseEstimator, Classifier)):
        clf = {'clf': clf}

    # if clf is a string only, create dict with only the type specified
    if isinstance(clf, str):
        assert clf in __IMPLEMENTED_CLFS, "clf not implemented (yet. Make an issue;) )"
        clf = {'clf_type': clf, 'config': {}}

    assert isinstance(clf, dict), "Wrong data format of classifier..."

    if isinstance(clf.get('n_cpu'), int) and n_cpu is None:
        n_cpu = clf['n_cpu']

    # Warning if n_cpu of clf is bigger then n_cpu, but only if not None
    suppress_cpu_warning = False
    if n_cpu is None:
        suppress_cpu_warning = True
    n_cpu = meta_config.get_n_cpu(n_cpu)

    # if input is dict containing a clf, make sure it's a Sklearn one
    if len(clf) == 1 and isinstance(clf.values()[0], (BaseEstimator, Classifier)):
        key, value = clf.popitem()
        clf['name'] = key
        clf['clf'] = value
    if isinstance(clf.get('clf'), (BaseEstimator, Classifier)):
        classifier = clf['clf']
        clf_type = None
        if not isinstance(classifier, Classifier):
            classifier = SklearnClassifier(clf=classifier)
        output['clf'] = classifier

        # Test which classifier it is and get parallel_profile
        if isinstance(classifier, XGBoostClassifier):
            n_cpu_clf = classifier.nthreads
            clf_type = 'xgb'
        elif isinstance(classifier, TheanetsClassifier):
            n_cpu_clf = 1
            clf_type = 'nn'
        elif isinstance(classifier, TMVAClassifier):
            n_cpu_clf = 1
            clf_type = 'tmva'
        elif isinstance(classifier, SklearnClassifier):
            sub_clf = classifier.clf
            if isinstance(sub_clf, RandomForestClassifier):
                n_cpu_clf = sub_clf.n_jobs
                clf_type = 'rdf'
            elif isinstance(sub_clf, AdaBoostClassifier):
                n_cpu_clf = 1
                clf_type = 'ada'
            elif isinstance(sub_clf, GradientBoostingClassifier):
                n_cpu_clf = 1
                clf_type = 'gb'
            elif isinstance(sub_clf, KNeighborsClassifier):
                n_cpu_clf = 1
                clf_type = 'knn'
        else:
            n_cpu_clf = 1

        if n_cpu_clf > n_cpu and not suppress_cpu_warning:
            logger.warning("n_cpu specified at make_clf() for clf < n_cpu of clf \
                            given! is that what you want?")
        n_cpu = max(int(n_cpu / n_cpu_clf), 1)
        if n_cpu > 1:
            output['n_cpu'] = n_cpu_clf
            output['parallel_profile'] = 'threads-' + str(n_cpu)
        else:
            output['n_cpu'] = n_cpu_clf
            output['parallel_profile'] = None
        clf_name = meta_config.DEFAULT_CLF_NAME.get(clf_type, "classifier")
        output['name'] = clf.get('name', clf_name)

    # If we do not have a classifier, we have a config dict and need to create a clf
    else:
        # find the clf_type and make sure it's an implemented one
        if 'clf_type' not in clf:
            for imp_clf in __IMPLEMENTED_CLFS:
                if imp_clf in clf:
                    clf['clf_type'] = imp_clf
                    clf['config'] = clf[imp_clf]
        if 'clf_type' not in clf:
            raise ValueError("Invalid classifier, not implemented")
        if 'name' not in clf:
            clf['name'] = clf['clf_type']
        default_clf = dict(
            clf_type=clf['clf_type'],
            name=meta_config.DEFAULT_CLF_NAME[clf['clf_type']],
            config=meta_config.DEFAULT_CLF_CONFIG[clf['clf_type']],
        )

        clf = dict(default_clf, **clf)

        if clf['clf_type'] == 'xgb':
            # update config dict with parallel-variables and random state
            clf['config'].update(dict(nthreads=n_cpu, random_state=meta_config.randint()))
            clf_tmp = XGBoostClassifier(**clf.get('config'))
        elif clf['clf_type'] == 'tmva':
            serial_clf = True
            clf_tmp = TMVAClassifier(**clf.get('config'))
        elif clf['clf_type'] == 'gb':
            serial_clf = True
            clf_tmp = SklearnClassifier(GradientBoostingClassifier(**clf.get('config')))
        elif clf['clf_type'] == 'rdf':
            clf['config'].update(dict(n_jobs=n_cpu, random_state=meta_config.randint()))
            clf_tmp = SklearnClassifier(RandomForestClassifier(**clf.get('config')))
        elif clf['clf_type'] == 'ada':
            serial_clf = True
            clf['config'].update(dict(random_state=meta_config.randint()))
            clf_tmp = SklearnClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
                random_state=meta_config.randint()), **clf.get('config')))
        elif clf['clf_type'] == 'knn':
            clf['config'].update(dict(random_state=meta_config.randint(), n_jobs=n_cpu))
            clf_tmp = SklearnClassifier(KNeighborsClassifier(**clf.get('config')))
        elif clf['clf_type'] == 'rdf':
            clf['config'].update(dict(n_jobs=n_cpu, random_state=meta_config.randint()))
            clf_tmp = SklearnClassifier(RandomForestClassifier(**clf.get('config')))
        elif clf['clf_type'] == 'nn':
            serial_clf = meta_config.use_gpu
            clf['config'].update(dict(random_state=meta_config.randint()))
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


def backward_feature_elimination(original_data, target_data=None, features=None,
                                 clf='xgb', n_folds=10, max_feature_elimination=None,
                                 max_difference_to_best=0.08, keep_features=None,
                                 take_target_from_data=False):
    """Train and score on each feature subset, eliminating features backwards.

    To know, which features make a big impact on the training of the clf and
    which don't, there are several techniques to find out. The most reliable,
    but also cost-intensive one, is the recursive backward feature elimination.
    A classifier gets trained first on all the features and is validated with
    the KFold-technique and the ROC AUC. Then, a feature is removed and the
    classifier is trained and tested again. This is done for all features once.
    The feature where the auc drops the least is then removed and the next round
    starts from the beginning but with one feature less.

    The function ends either if:

    - no features are left
    - max_feature_elimination features have been eliminated
    - the time limit max_feature_elimination is reached
    - the difference between the most useless features auc and the best
      (the run done with all features in the beginning) is higher then
      max_difference_to_best

    Parameters
    ----------
    original_data : HEPDataStorage
        The original data
    target_data : HEPDataStorage
        The target data
    features : list(str, str, str,...)
        List of strings containing the features/columns to be used for the
        hyper-optimization or feature selection.
    clf : str {'xgb, 'rdf, 'erf', 'gb', 'ada', 'nn'} or config-dict
        For possible options, see also :py:func:`~raredecay.ml_analysis.make_clf()`
    n_folds : int > 1
        How many folds you want to split your data in when doing KFold-splits
        to measure the performance of the classifier.
    max_feature_elimination : int >= 1 or str "hhhh:mm"
        How many features should be maximal eliminated before it stopps or
        how much time it can take (approximately) to do the elimination.
        If the time runs out before other criterias are true (no features left,
        max_difference to high...), it just returns the results so far.
    max_difference_to_best : float
        The maximum difference between the "least worst" features auc and the best
        (usually the one with all features) auc before it stopps.

        In other words, it only eliminates features until the elimination would
        lead to a roc auc lower by max_difference_to_best then the roc auc
        with all features (= highest roc auc).
    keep_features:
        A list of features that won't be eliminated. The algorithm does not
        test the metric if that feature were removed. This saves
        quite some time.
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
    direction = 'backward'
    keep_features = [] if keep_features is None else data_tools.to_list(keep_features)
    output = {}
    start_time = -1  # means: no time measurement on the way
    available_time = 1

    # start timer if time-limit is given
    if isinstance(max_feature_elimination, str):
        max_feature_elimination = max_feature_elimination.split(":")
        assert len(max_feature_elimination) == 2, "Wrong time-format. Has to be 'hhh...hhh:mm' "
        available_time = (3600 * int(max_feature_elimination[0]) +
                          60 * int(max_feature_elimination[1]))
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
    features = list(set(features + keep_features))
    assert features != [], "No features for optimization found"

    # initialize data
    data, label, weights = _make_data(original_data, target_data, features=features,
                                      target_from_data=take_target_from_data)

    # initialize clf and parallel_profile
    clf_dict = make_clf(clf=clf, n_cpu=meta_config.n_cpu_max)
    clf = clf_dict['clf']
    clf_name = clf_dict['name']
    parallel_profile = clf_dict['parallel_profile']

# ==============================================================================
# start backward feature elimination
# ==============================================================================
    selected_features = copy.deepcopy(features)  # explicit is better than implicit
    selected_features = [feature for feature in selected_features if feature not in keep_features]

    assert len(selected_features) > 1, "Need more then one feature to perform feature selection"

    # starting feature selection
    out.add_output(["Performing feature selection with the classifier",
                    clf_name, "of the features", features],
                   title="Feature selection: Recursive backward elimination")
    original_clf = FoldingClassifier(clf, n_folds=n_folds,
                                     stratified=meta_config.use_stratified_folding,
                                     parallel_profile=parallel_profile)

    # "loop-initialization", get score for all features
    roc_auc = OrderedDict({})
    collected_scores = {feature: [] for feature in selected_features}
    if direction == 'backward':
        clf = copy.deepcopy(original_clf)  # required, feature attribute can not be changed somehow
        clf.fit(data[features], label, weights)
        report = clf.test_on(data[features], label, weights)
        max_auc = report.compute_metric(metrics.RocAuc()).values()[0]
        roc_auc = OrderedDict({'all features': round(max_auc, 4)})
        out.save_fig(figure="feature importance " + str(clf_name), importance=2, **save_fig_cfg)
        # HACK: temp_plotter1 is used to set the plot.new_plot to False,
        # which is set to True (unfortunately) in the init of GridPlot
        temp_plotter1 = report.feature_importance_shuffling()
        temp_plotter1.new_plot = False
        temp_plotter1.plot(title="Feature importance shuffling of " + str(clf_name))
        # HACK END
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

    iterations = 0  # for the timing
    # do-while python-style (with if-break inside)
    while n_to_eliminate > 0:

        # initialize variable
        difference = 1  # a surely big initialisation
        n_to_eliminate -= 1
        n_features_left = len(selected_features)
        collected_scores['features_tot'].append(n_features_left)

        # iterate through the features and remove the ith each time
        for i, feature in enumerate(selected_features):
            iterations += 1
            clf = copy.deepcopy(original_clf)  # otherwise feature attribute trouble
            temp_features = copy.deepcopy(selected_features)
            del temp_features[i]  # remove ith feature for testing
            clf.fit(data[temp_features + keep_features], label, weights)
            report = clf.test_on(data[temp_features + keep_features], label, weights)
            temp_auc = report.compute_metric(metrics.RocAuc()).values()[0]
            collected_scores[feature].append(round(temp_auc, 4))
            # set time condition, extrapolate assuming same time for each iteration
            eet_next = (timeit.default_timer() - start_time) * (iterations + 1) / iterations
            if available_time < eet_next and start_time > 0:
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
    out.add_output(["The collected scores:\n"] +
                   [collected_scores[col] for col in collected_scores],
                   importance=3)
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
                        "max_feature_elimination was reached",
                        "(max number of eliminations or time limit)."],
                       subtitle="Feature selection results")
    # if all features were removed
    else:
        out.add_output(["Removed features and roc auc: ", roc_auc,
                        "All features removed, loop stopped removing because no feature was left"],
                       subtitle="Feature selection results")

    return output


def optimize_hyper_parameters(original_data, target_data=None, clf=None, features=None,
                              n_eval=1, n_checks=10, n_folds=10, generator_type='subgrid',
                              take_target_from_data=False, **kwargs):
    """Optimize the hyperparameters of a classifiers.

    Hyper-parameter optimization of a classifier is an important task.
    Two datasets are required as well as a clf (not an instance, a dict).
    For more information about which classifiers are valid, see also
    :py:func:`~raredecay.analysis.ml_analysis.make_clf()`.

    The optimization does not happen automatic but checks the hyper-parameter
    space provided. Every clf-parameter that is a list or numpy array is
    considered a point. The search-technique can be specified under
    *generator_type*.

    It is possible to set a time limit instead of a n_eval limit. It estimates
    the time needed for a run and extrapolates. This extrapolation is not too
    precise, it can be at *worst* plus approximately 20% of allowed run-time,


    Parameters
    ----------
    original_data : HEPDataStorage
        The original data
    target_data : HEPDataStorage
        The target data
    clf : config-dict
        For possible options, see also
        :py:func:`~raredecay.analysis.ml_analysis.make_clf()`.
        The difference is, for the feature you want to have optimised, use an
        iterable instead of a single value, e.g. 'n_estimators': [1, 2, 3, 4] etc.
    features : list(str, str, str,...)
        List of strings containing the features/columns to be used for the
        hyper-optimization.
    n_eval : int > 1 or str "hh...hh:mm"
        How many evaluations should be done; how many points in the
        hyperparameter-space should be tested. This can either be an integer,
        which then represents the number of evaluations done or a string in the
        format of "hours:minutes" (e.g. "3:25", "1569:01" (quite long...),
        "0:12"), which represents the approximat time it should take for the
        hyperparameter-search (**not** the exact upper limit)
    n_checks : 1 <= int <= n_folds
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
        |take_target_from_data_docstring|
    """
    # initialize variables and setting defaults
#    output = {}
#    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    clf_dict = make_clf(clf, n_cpu=meta_config.n_cpu_max, dict_only=True)
    config_clf = clf_dict['config']
    config_clf_cp = copy.deepcopy(config_clf)

    # Create parameter for clf and hyper-search
    if features is None:
        features = original_data.columns

    grid_param = {}
    # parameters which are by their nature a list, e.g. nn-layers
    list_param = ['layers', 'trainers']
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
        n_checks_tmp = 1  # time will be multiplied by actual n_checks

        # call hyper_optimization with parameters for "one" run and measure time
        out.add_output(data_out="", subtitle="Starting small test-run for time estimation.",
                       importance=2)
        # do-while loop
        clf_tmp = copy.deepcopy(clf)
        clf_tmp['config'] = config_clf_cp
        while True:
            start_timer = timeit.default_timer()
            optimize_hyper_parameters(original_data, target_data=target_data, clf=clf_tmp,
                                      n_eval=n_eval_tmp, n_folds=n_folds, n_checks=n_checks_tmp,
                                      features=features, generator_type=generator_type,
                                      take_target_from_data=take_target_from_data, time_test=True)
            elapsed_time = timeit.default_timer() - start_timer
            if elapsed_time > min_elapsed_time:
                break
            elif n_checks_tmp < n_checks:  # for small datasets, increase n_checks for testing
                n_checks_tmp = min(n_checks, np.ceil(min_elapsed_time / elapsed_time))
            else:  # if time to small, increase n_rounds
                n_eval_tmp *= np.ceil(min_elapsed_time / elapsed_time)

        elapsed_time *= np.ceil(float(n_checks) / n_checks_tmp)  # time for "one round"
        test_time = timeit.default_timer() - start_timer_test
        n_eval = ((int((available_time * 0.98 - test_time) / elapsed_time)) *
                  int(round(n_eval_tmp)))  # we did just one
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
             clf='xgb', extended_report=False, get_predictions=False,
             plot_title=None, curve_name=None, weights_ratio=0,
             importance=3, plot_importance=3,
             target_from_data=False, **kwargs):
    """Training and/or testing a classifier or kfolded predictions.

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
    - Get a ROC curve of two datasets with K-Folding.
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
    clf : classifier, see :py:func:`~raredecay.analysis.ml_analysis.make_clf()`
        The classifier to be used for the training and predicting. It can also
        be a pretrained classifier as argument.
    extended_report : boolean
        If True, make extended reports on the classifier as well as on the data,
        including feature correlation, feature importance etc.
    get_predictions : boolean
        If True, return a dictionary containing the prediction probabilities, the
        true y-values, weights and more. Have a look at the return values.
    plot_title : str
        A part of the title of the plots and general name of the call. Will
        also be printed in the output to identify with the intention this
        function was called.
    curve_name : str
        A labeling for the plotted data.
    weights_ratio : int >= 0
        The ratio of the weights, actually the class-weights.
    importance : |importance_type|
        |importance_docstring|
    plot_importance : |plot_importance_type|
        |plot_importance_docstring|
    target_from_data : boolean
        |take_target_from_data_docstring|

    additional kwargs arguments :
        original_test_weights : pandas Series
            Weights for the test sample if you don't want to use the same
            weights as in the training
        target_test_weights : pandas Series
            Weights for the test sample if you don't want to use the same
            weights as in the training

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
    VALID_KWARGS = ['original_test_weights', 'target_test_weights']

    # initialize variables and data
    save_fig_cfg = dict(meta_config.DEFAULT_SAVE_FIG, **cfg.save_fig_cfg)
    predictions = {}
    make_plot = True  # used if no validation
    valid_input = set(kwargs).issubset(VALID_KWARGS)
    if not valid_input:
        raise ValueError("Invalid kwargs:" + str([k for k in kwargs if k not in VALID_KWARGS]))

    plot_title = "classify" if plot_title is None else plot_title

    if original_data is not None:
        data, label, weights = _make_data(original_data, target_data, features=features,
                                          weights_ratio=weights_ratio,
                                          target_from_data=target_from_data)
        data_name = original_data.name
        if target_data is not None:
            data_name += " and " + target_data.name

    clf_dict = make_clf(clf, n_cpu=-1)
    clf = clf_dict['clf']
    clf_name = clf_dict.pop('name')
    parallel_profile = clf_dict.get('parallel_profile')

    if isinstance(validation, (float, int, long)) and validation > 1:
        if 'original_test_weights' in kwargs or 'target_test_weights' in kwargs:

            if 'original_test_weights' in kwargs:
                temp_original_weights = original_data.get_weights()
                original_data.set_weights(kwargs.get('original_test_weights'))
            if 'target_test_weights' in kwargs:
                temp_target_weights = target_data.get_weights()
                target_data.set_weights(kwargs.get('target_test_weights'))
            test_weights = original_data.get_weights(second_storage=target_data,
                                                     normalization=weights_ratio)
            if 'original_test_weights' in kwargs:
                original_data.set_weights(temp_original_weights)
            if 'target_test_weights' in kwargs:
                target_data.set_weights(temp_target_weights)

        else:
            test_weights = weights

        clf = FoldingClassifier(clf, n_folds=int(validation), parallel_profile=parallel_profile,
                                stratified=meta_config.use_stratified_folding)
        # folding-> same data for train and test
        lds_test = LabeledDataStorage(data=data, target=label, sample_weight=test_weights)

    elif isinstance(validation, data_storage.HEPDataStorage):
        lds_test = validation.get_LabeledDataStorage(columns=features)
    elif validation in (None, False):
        make_plot = False
        clf_score = None
    elif isinstance(validation, list) and len(validation) in (1, 2):
        data_val, target_val, weights_val = _make_data(validation[0], validation[1])
        lds_test = LabeledDataStorage(data=data_val, target=target_val, sample_weight=weights_val)
    else:
        raise ValueError("Validation method " + str(validation) + " not a valid choice")

    # train the classifier
    if original_data is not None:
        clf.fit(data, label, weights)
        # if error "1 not in list" or similar occurs: no valid targets (None?)

    # test the classifier
    if validation not in (None, False):
        report = ClassificationReport({clf_name: clf}, lds_test)
        test_classes = list(set(lds_test.get_targets()))
        n_classes = len(test_classes)
        if n_classes == 2:
            clf_score = round(report.compute_metric(metrics.RocAuc()).values()[0], 4)
            out.add_output(["ROC AUC of ", clf_name, ", ", curve_name, ": ", clf_score],
                           obj_separator="", subtitle="Report of " + plot_title,
                           importance=importance)
            plot_name = clf_name + ", AUC = " + str(clf_score)
            binary_test = True
            if get_predictions:
                # TODO: DRY, now WET
                y_true = lds_test.get_targets()
                y_pred = clf.predict(lds_test.get_data())
                y_pred_proba = clf.predict_proba(lds_test.get_data())
                predictions['y_proba'] = y_pred_proba
                predictions['y_pred'] = y_pred
                predictions['y_true'] = y_true
                predictions['weights'] = lds_test.get_weights(allow_nones=True)
                predictions['report'] = report

        elif n_classes == 1:
            # score returns accuracy; if only one label present, it is the same as recall
            y_true = lds_test.get_targets()
            y_pred = clf.predict(lds_test.get_data())

            if get_predictions:
                y_pred_proba = clf.predict_proba(lds_test.get_data())
                predictions['y_proba'] = y_pred_proba
                predictions['y_pred'] = y_pred
                predictions['y_true'] = y_true
                predictions['weights'] = lds_test.get_weights(allow_nones=True)
                predictions['report'] = report
            w_test = lds_test.get_weights()
            clf_score = clf.score(lds_test.get_data(), y_true, w_test)
            clf_score2 = accuracy_score(y_true=y_true, y_pred=y_pred)  # , sample_weight=w_test)
            class_rep = classification_report(y_true, y_pred, sample_weight=w_test)
            out.add_output(class_rep, section="Classification report " + clf_name,
                           importance=importance)
            out.add_output(["recall of", clf_name, ",", curve_name, ":", clf_score],
                           importance=importance)
            binary_test = False
            plot_name = clf_name + ", recall = " + str(clf_score)
            out.add_output(["accuracy NO WEIGHTS! (just for curiosity):", clf_name,
                            ",", curve_name, ":", clf_score2], importance=importance,
                           subtitle="Report of classify: " + str(plot_name))
        else:
            raise ValueError("Multi-label classification not supported")

    # plots

    if make_plot:  # if no validation is given, don't make plots
        if curve_name is not None:
            plot_name = curve_name + " " + plot_name
        report.prediction[plot_name] = report.prediction.pop(clf_name)
        report.estimators[plot_name] = report.estimators.pop(clf_name)

        if binary_test:
            out.save_fig(plot_title + " " + plot_name,
                         importance=plot_importance, **save_fig_cfg)
            report.roc(physics_notion=True).plot(title=plot_title + "\nROC curve of " +
                                                 clf_name + " on data:" +
                                                 data_name + "\nROC AUC = " + str(clf_score))
            plt.plot([0, 1], [1, 0], 'k--')  # the fifty-fifty line

            out.save_fig("Learning curve " + plot_name,
                         importance=plot_importance, **save_fig_cfg)
            report.learning_curve(metrics.RocAuc(), steps=1).plot(title="Learning curve of " +
                                                                  plot_name)
        else:
            pass
            # TODO: implement learning curve with tpr metric
#            out.save_fig(plt.figure("Learning curve" + plot_name),
#                         importance=plot_importance, **save_fig_cfg)
#            report.learning_curve(metrics., steps=1).plot(title="Learning curve of " + plot_name)
        if extended_report:
            if len(clf.features) > 1:
                out.save_fig(figure="Feature importance shuffling of " + plot_name,
                             importance=plot_importance)
                # HACK: temp_plotter1 is used to set the plot.new_plot to False,
                # which is set to True (unfortunately) in the init of GridPlot
                temp_plotter1 = report.feature_importance_shuffling()
                temp_plotter1.new_plot = False
                temp_plotter1.plot(title="Feature importance shuffling of " + plot_name)
                # HACK END
                out.save_fig(figure="Feature correlation matrix of " + plot_name,
                             importance=plot_importance)
                report.features_correlation_matrix().plot(title="Feature correlation matrix of " +
                                                          plot_name)
            label_dict = None if binary_test else {test_classes[0]: "validation data"}
            out.save_fig(figure="Predictions of " + plot_name, importance=plot_importance)
            report.prediction_pdf(plot_type='bar', labels_dict=label_dict).plot(
                title="Predictions of " + plot_name)

    if clf_score is None:
        return clf
    elif get_predictions:
        return clf, clf_score, predictions
    else:
        return clf, clf_score


def reweight_train(mc_data, real_data, columns=None,
                   reweighter='gb', reweight_saveas=None, meta_cfg=None,
                   weights_mc=None, weights_real=None):
    """Return a trained reweighter from a (mc/real) distribution comparison.

    | Reweighting a distribution is a "making them the same" by changing the
      weights of the bins (instead of 1) for each event. Mostly, and therefore
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
    mc_data : |hepds_type|
        The Monte-Carlo data to compare with the real data.
    real_data : |hepds_type|
        Same as *mc_data* but for the real data.
    columns : list of strings
        The columns/features/branches you want to use for the reweighting.
    reweighter : {'gb', 'bins'}
        Specify which reweighter to be used.

        - **gb**: The GradientBoosted Reweighter from REP,
          :func:`~hep_ml.reweight.GBReweighter`
        - **bins**: The simple bins reweighter from REP,
          :func:`~hep_ml.reweight.BinsReweighter`
    reweight_saveas : string
        To save a trained reweighter in addition to return it. The value
        is the filepath + name.
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

    if reweighter not in __REWEIGHT_MODE:
        raise ValueError("Reweighter invalid: " + reweighter)

    reweighter = __REWEIGHT_MODE.get(reweighter.lower())
    reweighter += 'Reweighter'

    # logging and writing output
    msg = ["Reweighter:", reweighter, "with config:", meta_cfg]
    logger.info(msg)

    out.add_output(msg + ["\nData used:\n", mc_data.name, " and ",
                          real_data.name, "\ncolumns used for the reweighter training:\n",
                          columns], section="Training the reweighter", obj_separator=" ")

    if columns is None:
        # use the intesection of both colomns
        common_cols = set(mc_data.columns)
        common_cols.intersection_update(real_data.columns)
        columns = list(common_cols)
        if columns != mc_data.columns or columns != real_data.columns:
            logger.warning("No columns specified for reweighting, took intersection" +
                           " of both dataset, as it's columns are not equal." +
                           "\nTherefore some columns were not used!")
            meta_config.warning_occured()

    # create data
    mc_data, _t, mc_weights = _make_data(mc_data, features=columns,
                                         weights_original=weights_mc)
    real_data, _t, real_weights = _make_data(real_data, features=columns,
                                             weights_original=weights_real)
    del _t

    # train the reweighter
    if meta_cfg is None:
        meta_cfg = {}

    if reweighter == "GBReweighter":
        reweighter = hep_ml.reweight.GBReweighter(**meta_cfg)
    elif reweighter == "BinsReweighter":
        reweighter = hep_ml.reweight.BinsReweighter(**meta_cfg)
    reweighter.fit(original=mc_data, target=real_data,
                   original_weight=mc_weights, target_weight=real_weights)
    return data_tools.adv_return(reweighter, save_name=reweight_saveas)


def reweight_weights(reweight_data, reweighter_trained, columns=None,
                     normalize=True, add_weights_to_data=True):
    """Apply reweighter to the data and (add +) return the weights.

    Can be seen as a wrapper for the
    :py:func:`~hep_ml.reweight.GBReweighter.predict_weights` method.
    Additional functionality:
     * Takes a trained reweighter as argument, but can also unpickle one
       from a file.

    Parameters
    ----------
    reweight_data : |hepds_type|
        The data for which the reweights are to be predicted.
    reweighter_trained : (pickled) reweighter (*from hep_ml*)
        The trained reweighter, which predicts the new weights.
    columns : list(str, str, str,...)
        The columns to use for the reweighting.
    normalize : boolean or int
        If True, the weights will be normalized (scaled) to the value of
        normalize.
    add_weights_to_data : boolean
        If set to False, the weights will only be returned and not updated in
        the data (*HEPDataStorage*). If you want to use the data later on
        in the script with the new weights, set this value to True.

    Returns
    ------
    out : :py:class:`~pd.Series`
        Return an instance of pandas Series of shape [n_samples] containing the
        new weights.
    """
    normalize = 1 if normalize is True else normalize

    reweighter_trained = data_tools.try_unpickle(reweighter_trained)
    if columns is None:
        columns = reweighter_trained.columns
#    new_weights = reweighter_trained.predict_weights(reweight_data.pandasDF(),
    new_weights = reweighter_trained.predict_weights(reweight_data.pandasDF(columns=columns),
                                                     original_weight=reweight_data.get_weights())

    # write to output
    out.add_output(["Using the reweighter:\n", reweighter_trained, "\n to reweight ",
                    reweight_data.name], obj_separator="")

    if isinstance(normalize, (int, float)) and not isinstance(normalize, bool):
        new_weights *= new_weights.size / new_weights.sum() * normalize
    new_weights = pd.Series(new_weights, index=reweight_data.index)
    if add_weights_to_data:
        reweight_data.set_weights(new_weights)
    return new_weights


def reweight_Kfold(mc_data, real_data, columns=None, n_folds=10,
                   reweighter='gb', meta_cfg=None, n_reweights=1,
                   score_columns=None, score_clf='xgb',
                   add_weights_to_data=True, mcreweighted_as_real_score=False):
    """Kfold reweight the data by "itself" for *scoring* and hyper-parameters.

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
    mc_data : |hepds_type|
        The Monte-Carlo data, which has to be "fitted" to the real data.
    real_data : |hepds_type|
        Same as *mc_data* but for the real data.
    columns : list of strings
        The columns/features/branches you want to use for the reweighting.
    n_folds : int >= 1
        The number of folds to split the data. Usually, the more folds the
        "better" the reweighting (especially for small datasets).
        If n_folds = 1, the data will be reweighted directly and the benefit
        of Kfolds and the unbiasing *disappears*

    reweighter : {'gb', 'bins'}
        Specify which reweighter to use.
        - **gb**: GradientBoosted Reweighter from REP
        - **bins**: Binned Reweighter from REP
    meta_cfg : dict
        Contains the parameters for the bins/gb-reweighter. See also
        :func:`~hep_ml.reweight.BinsReweighter` and
        :func:`~hep_ml.reweight.GBReweighter`.
    n_reweights : int
        As the reweighting often yields different weights depending on random
        parameters like the splitting of the data, the new weights can be
        produced by taking the average of the weights over many reweighting
        runs. n_reweights is the number of reweight runs to average over.
    score_columns : list(str, str, str,...)
        The columns to use for the scoring. It is often a good idea to use
        different (and more) columns for the scoring then for the reweighting
        itself. A good idea is to use the same columns as for the selection
        later on.
    score_clf : clf or clf-dict or str
        The classifier to be used for the scoring.
        Has to be a valid argument to
        :py:func:`~raredecay.analysis.ml_analysis.make_clf`.
    add_weights_to_data : boolean
        If True, the new weights will be added (in place) to the mc data and
        returned. Otherwise, the weights will only be returned.
    mcreweighted_as_real_score : boolean or str
        If a string, it has to be an implemented classifier in *classify*.
        If true, the default ('xgb' most probably) will be used.
        |
        If not False, calculate and print the score. This scoring is based on a
        clf, which was trained on the not reweighted mc and real data and
        tested on the reweighted mc, and then predicts how many it "thinks"
        are real datapoints.
        |
        Intuitively, a classifiers learns to distinguish between mc and real
        and then classifies mc reweighted data labeled as real; he says, how
        "real" the reweighted data looks like. So a higher score is better.
        Drawback of this method is, it is completely blind to over-fitting
        of the reweighter. To get a relation, the classifier also predicts
        the mc (which should be an under limit) as well as the real data
        (which should be an upper limit).
        |
        Even dough this scoring sais not a lot about how well the reweighting
        worked, we can say, that if the score is higher than the real one,
        it has somehow over-fitted (if a classifier cannot classify, say,
        more than 70% of the real data as real, it should not be able to
        classify more than 70% of the reweighted mc as real. Reweighted mc
        should not "look more real" than real data)

    Return
    ------
    out : :py:class:`~pd.Series`
        Return the new weights.

    """
    output = {}
    out.add_output(["Doing reweighting_Kfold with ", n_folds, " folds"],
                   title="Reweighting Kfold", obj_separator="")
    # create variables
    assert n_folds >= 1 and isinstance(n_folds, int), \
        "n_folds has to be >= 1, its currently" + str(n_folds)
    assert isinstance(mc_data, data_storage.HEPDataStorage), \
        "wrong data type. Has to be HEPDataStorage, is currently" + str(type(mc_data))
    assert isinstance(real_data, data_storage.HEPDataStorage), \
        "wrong data type. Has to be HEPDataStorage, is currently" + str(type(real_data))

    new_weights_tot = pd.Series(np.zeros(len(mc_data)), index=mc_data.index)
    if mcreweighted_as_real_score:
        scores = np.zeros(n_folds)
        score_min = np.zeros(n_folds)
        score_max = np.zeros(n_folds)
    if not add_weights_to_data:
        old_mc_tot_weights = mc_data.get_weights()

    for run in range(n_reweights):
        new_weights_all = []
        new_weights_index = []

        # split data to folds and loop over them
        mc_data.make_folds(n_folds=n_folds)
        real_data.make_folds(n_folds=n_folds)
        logger.info("Data created, starting folding of run " + str(run) +
                    " of total " + str(n_reweights))

        for fold in range(n_folds):

            # create train/test data
            if n_folds > 1:
                train_real, test_real = real_data.get_fold(fold)
                train_mc, test_mc = mc_data.get_fold(fold)
            else:
                train_real = test_real = real_data.get_fold(fold)
                train_mc = test_mc = mc_data

            if mcreweighted_as_real_score:
                old_mc_weights = test_mc.get_weights()

            # plot the first fold as example (the first one surely exists)
            plot_importance1 = 3 if fold == 0 else 1
            if n_folds > 1 and plot_importance1 > 1 and run == 0:
                train_real.plot(figure="Reweighter trainer, example, fold " + str(fold),
                                importance=plot_importance1)
                train_mc.plot(figure="Reweighter trainer, example, fold " + str(fold),
                              importance=plot_importance1)

            # train reweighter on training data
            reweighter_trained = reweight_train(mc_data=train_mc,
                                                real_data=train_real,
                                                columns=columns, reweighter=reweighter,
                                                meta_cfg=meta_cfg)
            logger.info("reweighting fold " + str(fold) + "finished of run" + str(run))

            new_weights = reweight_weights(reweight_data=test_mc, columns=columns,
                                           reweighter_trained=reweighter_trained,
                                           add_weights_to_data=True)  # fold only, not full data
            # plot one for example of the new weights
            logger.debug("Maximum of weights " + str(max(new_weights)) +
                         " of fold " + str(fold) + " of run " + str(run))
            if (n_folds > 1 and plot_importance1 > 1) or max(new_weights) > 50:
                out.save_fig("new weights of fold " + str(fold), importance=plot_importance1)
                plt.hist(new_weights, bins=40, log=True)

            if mcreweighted_as_real_score:
                # treat reweighted mc data as if it were real data target(1)
                test_mc.set_targets(1)
                train_mc.set_targets(0)
                train_real.set_targets(1)
                # train clf on real and mc and see where it classifies the reweighted mc
                clf, tmp_score = classify(train_mc, train_real, validation=test_mc,
                                          curve_name="mc reweighted as real",
                                          features=score_columns,
                                          plot_title="fold {} reweighted validation".format(fold),
                                          weights_ratio=1, clf=score_clf,
                                          importance=1, plot_importance=1)
                scores[fold] += tmp_score

    # Get the max and min for "calibration" of the possible score for the reweighted data by
    # passing in mc and label it as real (worst/min score) and real labeled as real (best/max)
                test_mc.set_weights(old_mc_weights)
                _t, tmp_score_min = classify(clf=clf, validation=test_mc,
                                             features=score_columns,
                                             curve_name="mc as real",
                                             # weights_ratio=1,
                                             importance=1, plot_importance=1)
                score_min[fold] += tmp_score_min
                test_real.set_targets(1)
                _t, tmp_score_max = classify(clf=clf, validation=test_real,
                                             features=score_columns,
                                             curve_name="real as real",
                                             # weights_ratio=1,
                                             importance=1, plot_importance=1)
                score_max[fold] += tmp_score_max
                del _t

            # collect all the new weights to get a really cross-validated reweighted dataset
            new_weights_all.append(new_weights)
            new_weights_index.append(test_mc.get_index())

            logger.info("fold " + str(fold) + "finished")
            # end of for-loop

        # concatenate weights and index
        if n_folds == 1:
            new_weights_all = np.array(new_weights_all)
            new_weights_index = np.array(new_weights_index)
        else:
            new_weights_all = np.concatenate(new_weights_all)
            new_weights_index = np.concatenate(new_weights_index)
        new_weights_tot += pd.Series(new_weights_all, index=new_weights_index)
        logger.debug("Maximum of accumulated weights: " + str(max(new_weights_tot)))

        out.save_fig(figure="New weights of run " + str(run), importance=3)
        hack_array = np.array(new_weights_all)
        plt.hist(hack_array, bins=30, log=True)
        plt.title("New weights of reweighting at end of run " + str(run))

    # after for loop for weights creation
    new_weights_tot /= n_reweights

    if add_weights_to_data:
        mc_data.set_weights(new_weights_tot)
    else:
        mc_data.set_weights(old_mc_tot_weights)

    out.save_fig(figure="New weights of total mc", importance=4)
    plt.hist(new_weights_tot, bins=30, log=True)
    plt.title("New weights of reweighting with Kfold")

    # create score
    if mcreweighted_as_real_score:
        scores /= n_reweights
        score_min /= n_reweights
        score_max /= n_reweights
        out.add_output("", subtitle="Kfold reweight report", importance=4,
                       section="Precision scores of classification on reweighted mc")
        score_list = [("Reweighted: ", scores, 'score_reweighted'),
                      ("mc as real (min): ", score_min, 'score_min'),
                      ("real as real (max): ", score_max, 'score_max')]

        for name, score, key in score_list:
            mean, std = round(np.mean(score), 4), round(np.std(score), 4)
            out.add_output(["Classify the target, average score " + name + str(mean) +
                            " +- " + str(std)], to_end=True, importance=4)
            output[key] = mean

    output['weights'] = new_weights_tot
    return output


def best_metric_cut(mc_data, real_data, prediction_branch, metric='precision',
                    plot_importance=3):
    """Find the best threshold cut for a given metric.

    Test the metric for every possible threshold cut and returns the highest
    value. Plots the metric versus cuts as well.

    Parameters
    ----------
    mc_data : |hepds_type|
        The MC data
    real_data : |hepds_type|
        The real data
    prediction_branch : str
        The branch name containing the predictions to test.
    metric : str |implemented_primitive_metrics| or simple metric
        Can be a valid string pointing to a metric or a simple metric taking
        only tpr and fpr: metric(tpr, fpr, weights=None)
    plot_importance : |plot_importance_type|
        |plot_importance_docstring|

    Return
    ------
    out : dict
        Return a dict containing the best threshold cut as well as the metric
        value. The keywords are:

            - **best_threshold_cut**: the best cut on the predictions
            - **best_metric**: the value of the metric when applying the best
              cut.
    """
    from rep.report.metrics import OptimalMetric

    from raredecay.tools.metrics import punzi_fom, precision_measure

    metric_name = metric
    if metric == 'punzi':
        metric = punzi_fom
    elif metric == 'precision':
        metric = precision_measure

    data, target, weights = mc_data.make_dataset(real_data, columns=prediction_branch)
    predictions = data.T.as_matrix()[0, :]
    predictions = np.transpose(np.array((1 - predictions, predictions)))
    metric_optimal = OptimalMetric(metric)

    best_cut, best_metric = metric_optimal.compute(y_true=target,
                                                   proba=data,
                                                   sample_weight=weights)
    out.figure(str(metric_name) + " vs cut", importance=plot_importance)
    title = "{0} vs cut of {1} and {2}".format(str(metric_name), real_data.name, mc_data.name)
    metric_optimal.plot_vs_cut(y_true=target, proba=data, sample_weight=weights).plot(title=title)

    best_metric = np.nan_to_num(best_metric)
    best_index = np.argmax(best_metric)
    output = {'best_threshold_cut': best_cut[best_index],
              'best_metric': best_metric[best_index]}

    return output

if __name__ == "main":
    print 'test'
