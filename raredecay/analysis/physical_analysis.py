# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:49:45 2016

@author: mayou

Contains the different run-modes for the machine-learning algorithms.
"""
from __future__ import division, absolute_import

import importlib

import raredecay.meta_config

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

    # import configuration-file
    cfg = importlib.import_module(raredecay.meta_config.run_config)

    # initialize
    from raredecay.globals_ import set_output_handler
    set_output_handler(internal=True)
    from raredecay.globals_ import out
    out.initialize(logger_cfg=cfg.logger_cfg, **cfg.OUTPUT_CFG)
    out.add_output(["config file used", str(raredecay.meta_config.run_config)],
                    section="Configuration", obj_separator=" : ", to_end=True)

    # create logger
    from raredecay.tools import dev_tool
    logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)

    out.make_me_a_logger()  # creates a logger inside of "out"
    out.add_output(cfg.run_message, title="Run: "+cfg.RUN_NAME, do_print=False,
                   subtitle="Comments about the run")

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
            scores.append(score)
            scores_mean.append(np.mean(score))
        scores_mean = np.array(scores_mean)
        out.add_output(["Score of several CVreweighting:", scores], to_end=True)
        out.add_output(["Score mean:", np.mean(scores), "+- (measurements, NOT mean)",
                        np.std(scores)], to_end=True)
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
    out.add_output(['score:', scores['score'], "+-", scores['score_std']], do_print=True,
                   title='Train similar report', to_end=True)
    if scores.get('score_max', False):
        out.add_output(['score max:', scores['score_max'], "+-", scores['score_max_std']],
                       do_print=True, to_end=True)

    return scores['score']  # may you want to take the mean of several scorings, as it
                            # may vary around +-0.02. Ask for implementation or make it
                            # by implementing it into the if-elif statement at the beginning


def clf_mayou(cfg, logger):
  """Test a setup of clf involving bagging and stacking"""
  pass


def _hyper_optimization_int(cfg, logger, out):
    """Intern call to hyper_optimization"""
    from raredecay.tools import data_tools, dev_tool, data_storage

    original_data = data_storage.HEPDataStorage(**cfg.data['hyper_original'])
    target_data = data_storage.HEPDataStorage(**cfg.data['hyper_target'])

    #original_data.plot()

    clf = cfg.hyper_cfg['optimize_clf']
    config_clf = getattr(cfg, 'cfg_' + clf)

    n_eval = cfg.hyper_cfg['n_evaluations']
    n_checks = cfg.hyper_cfg['n_fold_checks']
    n_folds = cfg.hyper_cfg['n_folds']
    generator_type = cfg.hyper_cfg.get('generator')
    optimize_features = cfg.hyper_cfg.get('optimize_features', False)
    features = cfg.opt_features

    hyper_optimization(original_data=original_data, target_data=target_data,
                       features=features, optimize_features=optimize_features,
                       clf=clf, config_clf=config_clf, n_eval=n_eval,
                       n_checks=n_checks, n_folds=n_folds, generator_type=generator_type)



def hyper_optimization(original_data, target_data, clf, config_clf, n_eval, features,
                       n_checks=10, n_folds=10, generator_type="subgrid", optimize_features=False):
    """Perform hyperparameter optimization in this module"""
    import raredecay.analysis.ml_analysis as ml_ana

    ml_ana.optimize_hyper_parameters(original_data, target_data, features=features,
                                     clf=clf, config_clf=config_clf,
                                     optimize_features=optimize_features,
                                     n_eval=n_eval, n_checks=n_checks, n_folds=n_folds,
                                     generator_type=generator_type)

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
             columns=None, make_plots=True, apply_weights=True):

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
    new_weights = ml_ana.reweight_Kfold(reweight_data_mc=mc_data, reweight_data_real=real_data,
                                        meta_cfg=reweight_cfg, columns=columns,
                                        reweighter=reweighter, mcreweighted_as_real_score=scoring,
                                        n_folds=n_folds, make_plot=make_plots)

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
                                   n_folds=n_folds, n_checks=n_folds, test_predictions=False,
                                   make_plots=make_plots)

    # We can of course also test the normal ROC curve. This is weak to overfitting
    # but anyway (if not overfitting) a nice measure. You insert two datasets
    # and do the normal cross-validation on it. It's quite a multi-purpose
    # function depending on what validation is. If it is an integer, it means:
    # do cross-validation with n(=validation) folds.
    tmp_, classify_score = ml_ana.classify(original_data=mc_data, target_data=real_data,
                                           validation=n_folds, make_plots=make_plots)

    # an example to add output with the most importand parameters. The first
    # one can also be a single object instead of a list. do_print means
    # printing it also to the console instead of only saving it to the output
    # file. To_end is sometimes quite useful, as it prints (and saves) the
    # arguments at the end of the file. So the important results are possibly
    # printed to the end
    out.add_output(['score:', scores['score'], "+-", scores['score_std']], do_print=True,
                   title='Train similar report', to_end=True)
    if scores.get('score_max', False):
        out.add_output(['score max:', scores['score_max'], "+-", scores['score_max_std']],
                       do_print=True, to_end=True)

    output['weights'] = new_weights
    output['train_similar'] = scores
    output['roc_auc'] = classify_score

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
    run(1)
