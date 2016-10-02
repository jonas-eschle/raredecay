# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:49:45 2016

@author: mayou

Contains the different run-modes for the machine-learning algorithms.
"""
from __future__ import division, absolute_import

import importlib

import raredecay.meta_config

from memory_profiler import profile

__CFG_PATH = 'raredecay.run_config.'
DEFAULT_CFG_FILE = dict(
    reweightCV=__CFG_PATH + 'reweight_cfg',
    reweight=__CFG_PATH + 'reweight_cfg',
    simple_plot=__CFG_PATH + 'simple_plot1_cfg',
    test=__CFG_PATH + 'reweight1_comparison_cfg',
    reweight_comparison=__CFG_PATH + 'reweight1_comparison_cfg',
    hyper_optimization=__CFG_PATH + 'classifier_cfg',
    rafael1=__CFG_PATH + 'rafael_cfg'
)


def run(run_mode, cfg_file=None):
    """select the right runmode from the parameter and run it"""

    if cfg_file is None:
        cfg_file = DEFAULT_CFG_FILE.get(run_mode, None)
        assert cfg_file is not None, "No (default) cfg-file found."
    raredecay.meta_config.run_config = cfg_file
    raredecay.meta_config.NO_PROMPT_ASSUME_YES = False
    raredecay.meta_config.PROMPT_FOR_COMMENT = True


    # import configuration-file
    cfg = importlib.import_module(raredecay.meta_config.run_config)

    # initialize
    from raredecay.globals_ import out
    out.initialize_save(logger_cfg=cfg.logger_cfg, **cfg.OUTPUT_CFG)
    out.add_output(["config file used", str(raredecay.meta_config.run_config)],
                    section="Configuration", obj_separator=" : ", to_end=True)

    # create logger
    from raredecay.tools import dev_tool
    logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)

    out.make_me_a_logger()  # creates a logger inside of "out"


#==============================================================================
# Run initialized, start physical analysis
#==============================================================================

    if run_mode == "test":
        test(cfg, logger)
    elif run_mode == "reweight_comparison":
        reweight_comparison(cfg, logger)
    elif run_mode == "simple_plot":
        simple_plot(cfg, logger)
    elif run_mode == "reweightCV":
        import numpy as np
        scores = []
        scores_mean = []
        for i in range(1):
            score = _reweightCV_int(cfg, logger, out)
#            scores.append(score)
#        out.add_output(["Score of several CVreweighting:", scores], to_end=True)
#        out.add_output(["Score mean:", np.mean(scores), "+- (measurements, NOT mean)",
#                        np.std(scores)], to_end=True)
    elif run_mode == "reweight":
        _reweight_int(cfg, logger)
    elif run_mode == "hyper_optimization":
        _hyper_optimization_int(cfg, logger, out)
    elif run_mode == 'rafael1':
        rafael1(cfg=cfg, logger=logger, out=out)
    else:
        raise ValueError("Runmode " + str(run_mode) + " not a valid choice")

#==============================================================================
# Run finished, finalize it
#==============================================================================
    out.finalize()

def test(cfg):
    """just a test-function"""
    print "empty test function"


def rafael1(cfg, logger, out):


    """Test reweighting with CV and get reports on the performance

    To find the optimal parameters for the reweighting (most of all for the
    gradient boosted reweighter) it is crucial to reweight and test in a
    cross-validated way. There are several "metrics" to test the reweighting.

    Parameters
    ----------
    cfg : python-file
        The configuration file
    logger : a python logger
        The logger to be used. Should not be changed actually
    out : instance of output class
        The right instance which is placed in the meta-config
    """

    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_storage, metrics

    out.add_output("Starting the run", title="Reweighting Cross-Validated")
    # initialize variables
    n_folds = cfg.reweight_cv_cfg['n_folds']
    n_checks = cfg.reweight_cv_cfg.get('n_checks', n_folds)

    # just some "administrative variables, irrelevant
    plot_all = cfg.reweight_cv_cfg['make_plot']
    make_plots = True if plot_all in (True, 'all') else False

    # initialize data
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))

    # do the Kfold reweighting. This reweights the data with Kfolding and returns
    # the weights. If add_weights_to_data is True, the weights will automatically be
    # added to the reweight_data_mc (or here, reweight_mc). To get an estimate
    # wheter it has over-fitted, you can get the mcreweighted_as_real_score.
    # This trains a clf on mc/real and tests it on mc, mc reweighted, real
    # but both labeled with the same target as the real data in training
    # The mc reweighted score should therefore lie in between the mc and the
    # real score.
    ml_ana.reweight_Kfold(reweight_data_mc=reweight_mc, reweight_data_real=reweight_real,
                          meta_cfg=cfg.reweight_meta_cfg, columns=cfg.reweight_branches,
                          reweighter=cfg.reweight_cfg.get('reweighter', 'gb'),
                          mcreweighted_as_real_score=True, n_folds=n_folds, make_plot=make_plots)

    # To get a good estimation for the reweighting quality, the
    # train_similar score can be used. Its the one with training on
    # mc reweighted/real and test on real, quite robust.
    # Test_max is nice to know too even dough it can also be set to False if
    # testing the same distribution over and over again, as it is the same for
    # the same distributions (actually, it's just doing the score without the
    # weights).
    # test_predictions is an additional score I tried but so far I is not
    # reliable or understandable at all. The output, the scores dictionary,
    # is better described in the docs of the train_similar
    scores = metrics.train_similar(mc_data=reweight_mc, real_data=reweight_real, test_max=True,
                                   n_folds=n_folds, n_checks=n_checks, test_predictions=False,
                                   make_plots=make_plots)

    # We can of course also test the normal ROC curve. This is weak to overfitting
    # but anyway (if not overfitting) a nice measure. You insert two datasets
    # and do the normal cross-validation on it. It's quite a multi-purpose
    # function depending on what validation is. If it is an integer, it means:
    # do cross-validation with n(=validation) folds.
    ml_ana.classify(original_data=reweight_mc, target_data=reweight_real,
                    validation=10, make_plots=make_plots,
                    plot_title="",  # you can set an addition to the title. The
                                    # name of the data will be contained anyway
                    curve_name="mc reweighted/real")  # name of the curve; the legend

    # an example to add output with the most importand parameters. The first
    # one can also be a single object instead of a list. do_print means
    # printing it also to the console instead of only saving it to the output
    # file. To_end is sometimes quite useful, as it prints (and saves) the
    # arguments at the end of the file. So the important results are possibly
    # printed to the end
    out.add_output(['score:', scores['score'], "+-", scores['score_std']], importance=5,
                   title='Train similar report', to_end=True)
    if scores.get('score_max', False):
        out.add_output(['score max:', scores['score_max'], "+-", scores['score_max_std']],
                       importance=5, to_end=True)

    return scores['score']  # may you want to take the mean of several scorings, as it
                            # may vary around +-0.02. Ask for implementation or make it
                            # by implementing it into the if-elif statement at the beginning

#@profile
def clf_mayou(data1, data2, n_folds=3, n_base_clf=5):
    """Test a setup of clf involving bagging and stacking"""
    #import raredecay.analysis.ml_analysis as ml_ana
    import pandas as pd
    import copy

    from rep.estimators import SklearnClassifier, XGBoostClassifier, TMVAClassifier
    from rep.metaml.folding import FoldingClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
    from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, BaggingClassifier
    from rep.estimators.theanets import TheanetsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from rep.metaml.cache import CacheClassifier

    from rep.report.metrics import RocAuc

    from stacked_generalizer import StackedGeneralizer

    import rep.metaml.cache
    from rep.metaml._cache import CacheHelper
    rep.metaml.cache.cache_helper = CacheHelper('/home/mayou/cache', 100000)


#    data1.make_folds(n_folds)
#    data2.make_folds(n_folds)
    output = {}

    #for i in range(n_folds):
    xgb_clf = XGBoostClassifier(n_estimators=350, eta=0.1, max_depth=4, nthreads=3)
    xgb_folded = FoldingClassifier(base_estimator=xgb_clf, stratified=True, parallel_profile='threads-2')
    xgb_bagged = BaggingClassifier(base_estimator=xgb_folded, n_estimators=n_base_clf, bootstrap=False)
    xgb_bagged = SklearnClassifier(xgb_bagged)
    xgb_big_stacker = copy.deepcopy(xgb_bagged)
    xgb_bagged = CacheClassifier(name='xgb_bagged1', clf= xgb_bagged)


    xgb_single = XGBoostClassifier(n_estimators=350, eta=0.1, max_depth=4, nthreads=3)
    xgb_single = FoldingClassifier(base_estimator=xgb_single, stratified=True,
                                   n_folds=10, parallel_profile='threads-2')
    xgb_single = CacheClassifier(name='xgb_singled1', clf= xgb_single)


    rdf_clf = SklearnClassifier(RandomForestClassifier(n_estimators=300, n_jobs=3))
    rdf_folded = FoldingClassifier(base_estimator=rdf_clf, stratified=True, parallel_profile='threads-2')
    rdf_bagged = BaggingClassifier(base_estimator=rdf_folded, n_estimators=n_base_clf, bootstrap=False)
    rdf_bagged = SklearnClassifier(rdf_bagged)
    rdf_bagged = CacheClassifier(name='rdf_bagged1', clf=rdf_bagged)

    gb_clf = SklearnClassifier(GradientBoostingClassifier(n_estimators=50))
    gb_folded = FoldingClassifier(base_estimator=gb_clf, stratified=True, parallel_profile='threads-6')
    gb_bagged = BaggingClassifier(base_estimator=gb_folded, n_estimators=n_base_clf, bootstrap=False, n_jobs=5)
    gb_bagged = SklearnClassifier(gb_bagged)
    gb_bagged = CacheClassifier(name='gb_bagged1', clf=gb_bagged)

    nn_clf = TheanetsClassifier(layers=[300, 300], hidden_dropout=0.03,
                       trainers=[{'optimize': 'adagrad', 'patience': 5, 'learning_rate': 0.2, 'min_improvement': 0.1,
                       'momentum':0.4, 'nesterov':True, 'loss': 'xe'}])
    nn_folded = FoldingClassifier(base_estimator=nn_clf, stratified=True, parallel_profile=None)  #'threads-6')
    nn_bagged = BaggingClassifier(base_estimator=nn_folded, n_estimators=n_base_clf, bootstrap=False, n_jobs=1)
    nn_bagged = CacheClassifier(name='nn_bagged1', clf=nn_bagged)

    nn_single_clf = TheanetsClassifier(layers=[300, 300, 300], hidden_dropout=0.03,
                       trainers=[{'optimize': 'adagrad', 'patience': 5, 'learning_rate': 0.2, 'min_improvement': 0.1,
                       'momentum':0.4, 'nesterov':True, 'loss': 'xe'}])
    nn_single = FoldingClassifier(base_estimator=nn_single_clf, n_folds=3, stratified=True)
    nn_single = CacheClassifier(name='nn_single1', clf=nn_single)


    logit_stacker = SklearnClassifier(LogisticRegression(penalty='l2', solver='sag'))
    logit_stacker = FoldingClassifier(base_estimator=logit_stacker, n_folds=n_folds,
                                   stratified=True, parallel_profile='threads-6')
    logit_stacker = CacheClassifier(name='logit_stacker1', clf=logit_stacker)

    xgb_stacker = XGBoostClassifier(n_estimators=400, eta=0.1, max_depth=4, nthreads=8)
    #HACK
    xgb_stacker = xgb_big_stacker
    xgb_stacker = FoldingClassifier(base_estimator=xgb_stacker, n_folds=n_folds, random_state=42,
                                    stratified=True, parallel_profile='threads-6')
    xgb_stacker = CacheClassifier(name='xgb_stacker1', clf=xgb_stacker)


#        train1, test1 = data1.get_fold(i)
#        train2, test2 = data1.get_fold(i)
#
#        t_data, t_targets, t_weights =
    data, targets, weights = data1.make_dataset(data2, weights_ratio=1)

#    xgb_bagged.fit(data, targets, weights)
#    xgb_report = xgb_bagged.test_on(data, targets, weights)
#    xgb_report.roc(physics_notion=True).plot(new_plot=True, title="ROC AUC xgb_base classifier")
#    output['xgb_base'] = "roc auc:" + str(xgb_report.compute_metric(metric=RocAuc()))
#    xgb_proba = xgb_report.prediction['clf'][:, 1]
#    del xgb_bagged, xgb_folded, xgb_clf, xgb_report
#
#    xgb_single.fit(data, targets, weights)
#    xgb_report = xgb_single.test_on(data, targets, weights)
#    xgb_report.roc(physics_notion=True).plot(new_plot=True, title="ROC AUC xgb_single classifier")
#    output['xgb_single'] = "roc auc:" + str(xgb_report.compute_metric(metric=RocAuc()))
#    xgb_proba = xgb_report.prediction['clf'][:, 1]
#    del xgb_single, xgb_report

    nn_single.fit(data, targets, weights)
    nn_report = nn_single.test_on(data, targets, weights)
    nn_report.roc(physics_notion=True).plot(new_plot=True, title="ROC AUC nn_single classifier")
    output['nn_single'] = "roc auc:" + str(nn_report.compute_metric(metric=RocAuc()))
    nn_proba = nn_report.prediction['clf'][:, 1]
    del nn_single, nn_report

#    rdf_bagged.fit(data, targets, weights)
#    rdf_report = rdf_bagged.test_on(data, targets, weights)
#    rdf_report.roc(physics_notion=True).plot(new_plot=True, title="ROC AUC rdf_base classifier")
#    output['rdf_base'] = "roc auc:" + str(rdf_report.compute_metric(metric=RocAuc()))
#    rdf_proba = rdf_report.prediction['clf'][:, 1]
#    del rdf_bagged, rdf_clf, rdf_folded, rdf_report

#    gb_bagged.fit(data, targets, weights)
#    gb_report = gb_bagged.test_on(data, targets, weights)
#    gb_report.roc(physics_notion=True).plot(new_plot=True, title="ROC AUC gb_base classifier")
#    output['gb_base'] = "roc auc:" + str(gb_report.compute_metric(metric=RocAuc()))
#    gb_proba = gb_report.prediction['clf'][:, 1]
#    del gb_bagged, gb_clf, gb_folded, gb_report

#    nn_bagged.fit(data, targets, weights)
#    nn_report = nn_bagged.test_on(data, targets, weights)
#    nn_report.roc(physics_notion=True).plot(new_plot=True, title="ROC AUC nn_base classifier")
#    output['nn_base'] = "roc auc:" + str(nn_report.compute_metric(metric=RocAuc()))
#    nn_proba = nn_report.prediction['clf'][:, 1]
#    del nn_bagged, nn_clf, nn_folded, nn_report
#
#    base_predict = pd.DataFrame({'xgb': xgb_proba,
##                                 'rdf': rdf_proba,
#                                 #'gb': gb_proba,
#                                 'nn': nn_proba
#                                 })
#
#
#    xgb_stacker.fit(base_predict, targets, weights)
#    xgb_report = xgb_stacker.test_on(base_predict, targets, weights)
#    xgb_report.roc(physics_notion=True).plot(new_plot=True, title="ROC AUC xgb_stacked classifier")
#    output['stacker_xgb'] = "roc auc:" + str(xgb_report.compute_metric(metric=RocAuc()))
#    del xgb_stacker, xgb_report
#
#    logit_stacker.fit(base_predict, targets, weights)
#    logit_report = logit_stacker.test_on(base_predict, targets, weights)
#    logit_report.roc(physics_notion=True).plot(new_plot=True, title="ROC AUC logit_stacked classifier")
#    output['stacker_logit'] = "roc auc:" + str(logit_report.compute_metric(metric=RocAuc()))
#    del logit_stacker, logit_report

    print output



def _hyper_optimization_int(cfg, logger, out):
    """Intern call to hyper_optimization"""
    from raredecay.tools import data_tools, dev_tool, data_storage

    original_data = data_storage.HEPDataStorage(**cfg.data['hyper_original'])
    target_data = data_storage.HEPDataStorage(**cfg.data['hyper_target'])

#HACK
    clf_mayou(data1=original_data, data2=target_data)
    print "hack in use, physical analysis; _hyper_optimization_int"
    return
#HACK END
    #original_data.plot()

    clf = cfg.hyper_cfg['optimize_clf']
    config_clf = getattr(cfg, 'cfg_' + clf)

    n_eval = cfg.hyper_cfg['n_evaluations']
    n_checks = cfg.hyper_cfg['n_fold_checks']
    n_folds = cfg.hyper_cfg['n_folds']
    generator_type = cfg.hyper_cfg.get('generator')
    optimize_features = cfg.hyper_cfg.get('optimize_features', False)
    features = cfg.opt_features

    if optimize_features:
        hyper_optimization(original_data=original_data, target_data=target_data,
                           features=features, optimize_features=True,
                           clf=clf, config_clf=config_clf, n_eval=n_eval,
                           n_checks=n_checks, n_folds=n_folds, generator_type=generator_type)
    else:
        hyper_optimization(original_data=original_data, target_data=target_data,
                           features=features, optimize_features=False,
                           clf=clf, config_clf=config_clf, n_eval=n_eval,
                           n_checks=n_checks, n_folds=n_folds, generator_type=generator_type)


def feature_exploration(original_data, target_data, features=None, roc_auc=True):
    pass



def hyper_optimization(original_data, target_data, clf, config_clf, n_eval, features=None,
                       n_checks=10, n_folds=10, generator_type='subgrid', take_targets_from_data=False):
    """Perform hyperparameter optimization in this module"""
    import raredecay.analysis.ml_analysis as ml_ana

    ml_ana.optimize_hyper_parameters(original_data, target_data, features=features,
                                     clf=clf, config_clf=config_clf,
                                     optimize_features=False,
                                     n_eval=n_eval, n_checks=n_checks, n_folds=n_folds,
                                     generator_type=generator_type,
                                     take_target_from_data=take_targets_from_data)

    original_data.plot(figure="data comparison", title="data comparison", columns=features)
    target_data.plot(figure="data comparison", columns=features)


def add_branch_to_rootfile(root_data=None, new_branch=None, branch_name=None):
    """Add a branch to a given rootfile"""

    from raredecay.tools import data_tools
    from raredecay.globals_ import out

    out.add_output(["Adding", new_branch, "as", branch_name, "to",
                    root_data.get('filenames')], obj_separator=" ")

    data_tools.add_to_rootfile(root_data, new_branch=new_branch,
                               branch_name=branch_name)


def _reweight_int(cfg, logger, rootfile_to_add=None):
    """

    """

    from raredecay.tools import data_tools, data_storage
    from raredecay.globals_ import out
    import matplotlib.pyplot as plt

    out.add_output("Starting the run 'reweight'", title="Reweighting")

    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_apply = data_storage.HEPDataStorage(**cfg.data.get('reweight_apply'))

    reweight_apply.plot(figure="Data for reweights apply", title="Data before and after reweighting",
                        data_name="no weights")

    reweight_real.plot(figure="Data to train reweighter", data_name="before reweighting")
    reweight_mc.plot(figure="Data to train reweighter")

    print reweight(real_data=reweight_real, mc_data=reweight_mc, apply_data=reweight_apply,
             columns=cfg.reweight_branches, reweight_cfg=cfg.reweight_meta_cfg,
             reweighter='gb')

    # add weights to root TTree
    #add_branch_to_rootfile(cfg, logger, root_data=reweight_mc.get_rootdict(),
    #                       new_branch=new_weights, branch_name="weights_gb")


def reweight(real_data, mc_data, apply_data, reweighter='gb', reweight_cfg=None,
             columns=None, make_plots=True, apply_weights=True):
    """(Train a reweighter and) apply the reweighter do get new weights


    """
    import raredecay.analysis.ml_analysis as ml_ana

    from raredecay.globals_ import out

    import matplotlib.pyplot as plt

    output = {}

    if isinstance(reweighter, str):
        reweighter = ml_ana.reweight_train(reweight_data_mc=mc_data,
                                              reweight_data_real=real_data,
                                              columns=columns,
                                              meta_cfg=reweight_cfg,
                                              reweighter=reweighter)
    output['reweighter'] = reweighter
    output['weights'] = ml_ana.reweight_weights(reweight_data=apply_data,
                                                columns=columns,
                                                reweighter_trained=reweighter,
                                                add_weights_to_data=apply_weights)

    if make_plots:
        apply_data.plot(figure="Data for reweights apply", data_name="gb weights")
        out.save_fig(plt.figure("New weights"))
        plt.hist(output['weights'], bins=30, log=True)

    return output




def _reweightCV_int(cfg, logger, out):
    """Test reweighting with CV and get reports on the performance

    To find the optimal parameters for the reweighting (most of all for the
    gradient boosted reweighter) it is crucial to reweight and test in a
    cross-validated way. There are several "metrics" to test the reweighting.

    Parameters
    ----------
    cfg : python-file
        The configuration file
    logger : a python logger
        The logger to be used. Should not be changed actually
    out : instance of output class
        The right instance which is placed in the meta-config
    """

    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_tools, data_storage, metrics
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import copy

    from rep.estimators import XGBoostClassifier


    out.add_output("Starting the run 'reweightCV'", title="Reweighting Cross-Validated")
    # initialize variables
    n_folds = cfg.reweight_cv_cfg['n_folds']
    n_checks = cfg.reweight_cv_cfg.get('n_checks', n_folds)

    plot_all = cfg.reweight_cv_cfg['plot_all']
    make_plots = True if plot_all in (True, 'all') else False
#    score_gb = np.ones(n_checks)
#    score_min = np.ones(n_checks)
#    score_max = np.ones(n_checks)

    # initialize data
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))

    reweightCV(real_data=reweight_real, mc_data=reweight_mc,
               reweighter=cfg.reweight_cfg.get('reweighter', 'gb'),
               reweight_cfg=cfg.reweight_meta_cfg,
               columns=cfg.reweight_branches, make_plots=make_plots)



def reweightCV(real_data, mc_data, n_folds=10, reweighter='gb', reweight_cfg=None, scoring=True,
             n_folds_scoring=10, columns=None, make_plots=True, apply_weights=True):

    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import metrics
    from raredecay.globals_ import out


    output = {}
    # do the Kfold reweighting. This reweights the data with Kfolding and returns
    # the weights. If add_weights_to_data is True, the weights will automatically be
    # added to the reweight_data_mc (or here, reweight_mc). To get an estimate
    # wheter it has over-fitted, you can get the mcreweighted_as_real_score.
    # This trains a clf on mc/real and tests it on mc, mc reweighted, real
    # but both labeled with the same target as the real data in training
    # The mc reweighted score should therefore lie in between the mc and the
    # real score.
    if not apply_weights:
        old_weights = mc_data.get_weights()
    Kfold_output = ml_ana.reweight_Kfold(reweight_data_mc=mc_data, reweight_data_real=real_data,
                                        meta_cfg=reweight_cfg, columns=columns,
                                        reweighter=reweighter, mcreweighted_as_real_score=scoring,
                                        n_folds=n_folds, make_plot=make_plots)
    new_weights = Kfold_output.pop('weights')
    new_weights.sort_index()
    if scoring:
        output['mcreweighted_as_real_score'] = Kfold_output

    # To get a good estimation for the reweighting quality, the
    # train_similar score can be used. Its the one with training on
    # mc reweighted/real and test on real, quite robust.
    # Test_max is nice to know too even dough it can also be set to False if
    # testing the same distribution over and over again, as it is the same for
    # the same distributions (actually, it's just doing the score without the
    # weights).
    # test_predictions is an additional score I tried but so far I is not
    # reliable or understandable at all. The output, the scores dictionary,
    # is better described in the docs of the train_similar
    scores = metrics.train_similar(mc_data=mc_data, real_data=real_data, test_max=True,
                                   n_folds=n_folds_scoring, n_checks=n_folds_scoring,
                                   test_predictions=False, make_plots=make_plots)

    # We can of course also test the normal ROC curve. This is weak to overfitting
    # but anyway (if not overfitting) a nice measure. You insert two datasets
    # and do the normal cross-validation on it. It's quite a multi-purpose
    # function depending on what validation is. If it is an integer, it means:
    # do cross-validation with n(=validation) folds.
    tmp_, roc_auc_score = ml_ana.classify(original_data=mc_data, target_data=real_data,
                                           validation=n_folds_scoring, make_plots=make_plots)

    # an example to add output with the most importand parameters. The first
    # one can also be a single object instead of a list. do_print means
    # printing it also to the console instead of only saving it to the output
    # file. To_end is sometimes quite useful, as it prints (and saves) the
    # arguments at the end of the file. So the important results are possibly
    # printed to the end
    out.add_output(['ROC AUC score:', roc_auc_score], importance=5,
                   title='ROC AUC of mc reweighted/real KFold', to_end=True)
    out.add_output(['score:', scores['score'], "+-", scores['score_std']], importance=5,
                   title='Train similar report', to_end=True)
    if scores.get('score_max', False):
        out.add_output(['score max:', scores['score_max'], "+-", scores['score_max_std']],
                       importance=5, to_end=True)

    output['weights'] = new_weights
    output['train_similar'] = scores
    output['roc_auc'] = roc_auc_score
    if not apply_weights:
        mc_data.set_weights(old_weights)

    return output


def simple_plot(cfg, logger):

    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_storage
    from raredecay.globals_ import out
    import matplotlib.pyplot as plt

    mc_ee_original = data_storage.HEPDataStorage(**cfg.data.get('B2Kee_mc'))
    mc_jpsi_original = data_storage.HEPDataStorage(**cfg.data.get('B2KJpsi_mc'))
    mc_jpsi_cut = data_storage.HEPDataStorage(**cfg.data.get('B2KJpsi_mc_cut'))
    real_cut = data_storage.HEPDataStorage(**cfg.data.get('B2KpiLL_real_cut'))
    real_sweight = data_storage.HEPDataStorage(**cfg.data.get('B2KpiLL_real_cut_sweighted'))
    real_original = data_storage.HEPDataStorage(**cfg.data.get('B2KpiLL_real'))

#    real_cut.plot(figure="B2K1piLL data comparison: original-cut-sweighted (all normalized)",
#                  data_name="nEvents: " + str(len(real_cut)),
#                  title="B2K1piLL real data comparison: original-cut-sweighted")
#    real_original.plot(figure="B2K1piLL data comparison: original-cut-sweighted (all normalized)",
#                       data_name="nEvents: " + str(len(real_original)))
#    real_sweight.plot(figure="B2K1piLL data comparison: original-cut-sweighted (all normalized)",
#                      data_name="nEvents: " + str(len(real_sweight)))

#    mc_jpsi_original.plot(figure="B2K1Jpsi mc data comparison: original-cut (all normalized)",
#                          title="B2K1Jpsi mc data comparison: original-cut (all normalized)",
#                          data_name="nEvents: " + str(len(mc_jpsi_original)))
#    mc_jpsi_cut.plot(figure="B2K1Jpsi mc data comparison: original-cut (all normalized)",
#                          data_name="nEvents: " + str(len(mc_jpsi_cut)))


    real_cut.plot(figure="B2K1piLL CUT real vs mc (all normalized)",
                  data_name="nEvents: " + str(len(real_cut)),
                  title="B2K1piLL cut real vs mc comparison (all normalized)")
    mc_jpsi_cut.plot(figure="B2K1piLL CUT real vs mc (all normalized)",
                          data_name="nEvents: " + str(len(mc_jpsi_cut)))


    real_sweight.plot(figure="B2K1piLL sweighted real vs mc (all normalized)",
                  data_name="nEvents: " + str(len(real_cut)),
                  title="B2K1piLL sweighted real vs mc comparison (all normalized)")
    mc_jpsi_cut.plot(figure="B2K1piLL sweighted real vs mc (all normalized)",
                          data_name="nEvents: " + str(len(mc_jpsi_cut)))

#    mc_jpsi_original.plot(figure="B2K1piLL original real vs mc (all normalized)",
#                          title="B2K1piLL original real vs mc comparison (all normalized)",
#                          data_name="nEvents: " + str(len(mc_jpsi_original)))
#    real_original.plot(figure="B2K1piLL original real vs mc (all normalized)",
#                       data_name="nEvents: " + str(len(real_original)))

    mc_ee_original.plot(figure="B2K1ee mc original (normalized)",
                        title="B2K1ee mc original (normalized)",
                        data_name="nEvents: " + str(len(mc_ee_original)))


def reweight_comparison(cfg, logger):
    """

    """
    import matplotlib.pyplot as plt

    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_storage
    from raredecay.globals_ import out

    # make data
    logger.info("Start with gb reweighter")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    # TODO: remove
    gb_reweighter = ml_ana.reweight_train(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            columns=cfg.reweight_branches,
                                            meta_cfg=cfg.reweight_meta_cfg,
                                            **cfg.reweight_cfg)
    #gb_reweighter = 'gb_reweighter1.pickle'
    ml_ana.reweight_weights(reweight_mc, columns=cfg.reweight_branches,
                            reweighter_trained=gb_reweighter)
    reweight_mc.plot2Dscatter('B_PT', 'nTracks', figure=2)
    reweight_real.plot2Dscatter('B_PT', 'nTracks', figure=2, color='r')
    gb_roc_auc = ml_ana.data_ROC(original_data=reweight_mc,
                                 target_data=reweight_real, curve_name="GB reweighted",
                                 classifier='all')
    plot1 = reweight_mc.plot(figure="gradient boosted reweighting",
                     title="comparison real-target", data_name="self-reweighted", hist_settings={'bins':20})
    reweight_real.plot(figure="gradient boosted reweighting", hist_settings={'bins':20})
    out.save_fig(plot1, file_format=['png', 'svg'], to_pickle=False)
    out.save_fig(plt.figure("Weights bg reweighter"))
    plt.hist(reweight_mc.get_weights(), bins=20)
    plt.figure("weights from reweighting self")
    try:
        plt.hist([i for i in reweight_mc.get_weights() if i > -5], bins=200, log=True)
    except:
        pass

#==============================================================================
# predict new weights of unknown data
#==============================================================================
    reweight_apply = data_storage.HEPDataStorage(**cfg.data.get('reweight_apply'))

    reweight_apply.plot(figure="Data for reweights apply", title="Data before and after reweighting",
                        data_name="no weights")



    ml_ana.reweight_weights(reweight_data=reweight_apply, columns=cfg.reweight_branches,
                            reweighter_trained=gb_reweighter)
    reweight_apply.plot(figure="Data for reweights apply", data_name="gb weights")
    out.save_fig(plt.figure("New weights on new dataset"))
    plt.hist(reweight_apply.get_weights(), bins=30, log=True)


    reweight_apply.plot(figure="Comparison gb - bins reweighted", data_name="gb weights")

    print "mc weights sum", str(reweight_mc.get_weights().sum())
    print "real weights sum", str(reweight_real.get_weights().sum())
    #plt.show()
    return


    logger.info("Start with bins reweighter")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    reweight_apply = data_storage.HEPDataStorage(**cfg.data.get('reweight_apply'))

    logger.debug("plotted figure 2")
    bins_reweighter = ml_ana.reweight_train(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            reweighter='bins',
                                            columns=['B_PT', 'nTracks', 'nSPDHits'
                                            #, 'h1_TRACK_TCHI2NDOF'
                                            ],
                                            meta_cfg=cfg.reweight_meta_cfg_bins)
    #bins_reweighter = 'bins_reweighter1.pickle'
    ml_ana.reweight_weights(reweight_mc, bins_reweighter, columns=['B_PT', 'nTracks', 'nSPDHits'
                                            #, 'h1_TRACK_TCHI2NDOF'
                                            ],)
    ml_ana.reweight_weights(reweight_apply, bins_reweighter, columns=['B_PT', 'nTracks', 'nSPDHits'
                                            #, 'h1_TRACK_TCHI2NDOF'
                                            ],)
    reweight_mc.plot(figure="binned reweighting",
                     data_name="comparison real-target")
    reweight_real.plot(figure="binned reweighting")
    bins_roc_auc = ml_ana.data_ROC(original_data=reweight_mc,
                                   target_data=reweight_real, curve_name="Bins reweighted")

    reweight_apply.plot(figure="Comparison gb - bins reweighted", data_name="bins weights")
    # plt.show()



#    logger.debug("starting with original")
#    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
#    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
#    original_roc_auc = ml_ana.data_ROC(original_data=reweight_mc,
#                                       target_data=reweight_real, curve_name="Original weights")
#    reweight_mc.plot(figure="no reweighting",
#                     data_name="comparison real-target")
#    reweight_real.plot(figure="no reweighting")



# temporary:
if __name__ == '__main__':
    print "hello world 1"
    clf_mayou(1,2,3)
