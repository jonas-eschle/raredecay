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
    hyper_optimization=__CFG_PATH + 'classifier_cfg'
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
        reweightCV(cfg, logger)
    elif run_mode == "reweight":
        reweight(cfg, logger)
    elif run_mode == "hyper_optimization":
        hyper_optimization(cfg, logger)
    else:
        raise ValueError("Runmode " + str(run_mode) + " not a valid choice")

#==============================================================================
# Run finished, finalize it
#==============================================================================
    out.finalize()

def test(cfg):
    """just a test-function"""
    print "empty test function"


def hyper_optimization(cfg, logger):
    """Perform hyperparameter optimization in this module"""
    from raredecay.tools import data_tools, dev_tool, data_storage
    import raredecay.analysis.ml_analysis as ml_ana

    original_data = data_storage.HEPDataStorage(**cfg.data['hyper_original'])
    target_data = data_storage.HEPDataStorage(**cfg.data['hyper_target'])

    clf = cfg.hyper_cfg['optimize_clf']
    config_clf = getattr(cfg, 'cfg_' + clf)
    ml_ana.optimize_hyper_parameters(original_data, target_data, features=cfg.opt_features,
                                     clf=clf, config_clf=config_clf,
                                     optimize_features=cfg.hyper_cfg.get('optimize_features', False))

    original_data.plot(figure="data comparison", title="data comparison", columns=cfg.opt_features)
    target_data.plot(figure="data comparison", columns=cfg.opt_features)



def add_branch_to_rootfile(cfg, logger, root_data=None, new_branch=None,
                           branch_name=None):
    """Add a branch to a given rootfile"""

    from raredecay.tools import data_tools
    from raredecay.globals_ import out

    out.add_output(["Adding", new_branch, "as", branch_name, "to",
                    root_data.get('filenames')], obj_separator=" ")

    data_tools.add_to_rootfile(root_data, new_branch=new_branch,
                               branch_name=branch_name)


def reweight(cfg, logger, rootfile_to_add=None):
    """

    """
    import raredecay.analysis.ml_analysis as ml_ana
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

    gb_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            columns=cfg.reweight_branches,
                                            meta_cfg=cfg.reweight_meta_cfg,
                                            **cfg.reweight_cfg)
    new_weights = ml_ana.reweight_weights(reweight_data=reweight_apply,
                                          columns=cfg.reweight_branches,
                                          reweighter_trained=gb_reweighter)
    reweight_apply.plot(figure="Data for reweights apply", data_name="gb weights")
    out.save_fig(plt.figure("New weights"))
    plt.hist(reweight_apply.get_weights(), bins=30, log=True)

    new_weights = ml_ana.reweight_weights(reweight_data=reweight_mc, columns=cfg.reweight_branches,
                            reweighter_trained=gb_reweighter)
    reweight_real.plot(figure="Data self reweighted", data_name="gb weights")
    reweight_mc.plot(figure="Data self reweighted", data_name="after reweighting")

    # add weights to root TTree
    #add_branch_to_rootfile(cfg, logger, root_data=reweight_mc.get_rootdict(),
    #                       new_branch=new_weights, branch_name="weights_gb")



def reweightCV(cfg, logger, make_plot=True, minimal=False):
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
    make_plot : Boolean
        If True, the function will plot a lot of information like ROC curves,
        binned data and more. Otherwise, no plots will be made at all.
    minimal : Boolean
        If True, only a minimalistic check will be performed.

        The crossover reweighting creates the weights as always.
        The metric is the recall (or accuracy) of a classifier
          - trained on train real/ full mc reweighted
          - tested on leftout real test dataset
    """

    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_tools, data_storage
    from raredecay.globals_ import out
    import matplotlib.pyplot as plt
    import numpy as np
    import copy


    out.add_output("Starting the run 'reweightCV'", title="Reweighting Cross-Validated")
    # initialize variables
    n_folds = cfg.reweight_cv_cfg['n_folds']
    n_checks = min([cfg.reweight_cv_cfg['n_checks'], n_folds])
    score_gb = np.ones(n_checks)
    score_min = np.ones(n_checks)
    score_max = np.ones(n_checks)

    # initialize data
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_mc_reweighted = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))  # produces an error: copy.deepcopy(reweight_mc)

    reweight_real.make_folds(n_folds=n_folds)
    reweight_mc.make_folds(n_folds=n_folds)
    logger.info("Data created, starting folding")
    out.add_output(["Start reweighting cross-validated with", n_folds,
                    "fold of the data and do", n_checks, "checks on it."],
                    subtitle="cross-validation", obj_separator=" ")

    # iterate through different folds
    for fold in range(n_checks):
        # get train and test data
        train_real, test_real = reweight_real.get_fold(fold)
        train_mc, test_mc = reweight_mc.get_fold(fold)

        # plot the first fold as example (the first one surely exists)
        if ((fold == 0) or cfg.reweight_cv_cfg.get('plot_all', False)) and make_plot:
            train_real.plot(figure="Reweighter trainer, example, fold " + str(fold))
            train_mc.plot(figure="Reweighter trainer, example, fold " + str(fold))

        # train reweighter on training data
        reweighter = ml_ana.reweight_mc_real(meta_cfg=cfg.reweight_meta_cfg,
                                             reweight_data_mc=train_mc,
                                             reweight_data_real=train_real,
                                             columns=cfg.reweight_branches,
                                             **cfg.reweight_cfg)
        logger.info("reweighting fold " + str(fold) + "finished")

        # reweighter = ''  # load from pickle file

        # use reweighter on the test data_set
        new_weights = ml_ana.reweight_weights(test_mc, reweighter,
                                              columns=cfg.reweight_branches)

        # plot one for example of the new weights
        if (fold == 0) or cfg.reweight_cv_cfg.get('plot_all', False):
            plt.figure("new weights of fold " + str(fold))
            plt.hist(new_weights,bins=40, log=True)

        # treat reweighted mc data as if it were real data target(1)
        test_mc.set_targets(1)
        # train clf on real and mc and see where it classifies the reweighted mc
        clf, score_gb[fold] = ml_ana.classify(train_mc, train_real, validation=test_mc,
                                            curve_name="mc reweighted as real",
                                            plot_title="fold " + str(fold) + " reweighted validation",
                                            conv_ori_weights=False, weights_ratio=1)

        # Get the max and min for "calibration" of the possible score for the reweighted data by
        # passing in mc and label it as real (worst/min score) and real labeled as real (best/max)
        test_mc.set_weights(1)
        tmp_, score_min[fold] = ml_ana.classify(clf=clf, validation=test_mc, curve_name="mc as real")
        test_real.set_targets(1)
        tmp_, score_max[fold] = ml_ana.classify(clf=clf, validation=test_real, curve_name="real as real")


        # collect all the new weights to get a really cross-validated reweighted dataset
        if cfg.reweight_cv_cfg.get('total_roc', False) and (n_folds == n_checks):
            assert len(reweight_mc) == len(reweight_mc_reweighted), "Something bad happend somehow..."
            reweight_mc_reweighted.set_weights(new_weights, index=test_mc.get_index())
        logger.info("fold " + str(fold) + "finished")

    if cfg.reweight_cv_cfg.get('total_roc', False) and (n_folds == n_checks):
        logger.info("Starting data_ROC at the end of reweightCV")
        n_classify_checks = 3
        scores1 = np.ones(n_classify_checks)
        reweight_real.make_folds(n_folds=n_classify_checks)
        for fold in range(n_classify_checks):
            #create data
            real_train, real_test = reweight_real.get_fold(fold)
            real_test.set_targets(1)

            # train on reweighted and real data and classify test real data
            tmp_, score1 = ml_ana.classify(reweight_mc_reweighted, real_train, validation=real_test,
                                           plot_title="real/mc reweight trained, validate on real",
                                           weights_ratio=1)
            scores1[fold] = score1
        tmp_, score1_max = ml_ana.classify(reweight_mc, real_train, validation=real_test,
                                           plot_title="real/mc NOT reweight trained, validate on real",
                                           weights_ratio=1)
        out.add_output(["Score reweighted (recall, lower means better): ",
                        str(round(np.mean(scores1), 4)) + " +- " + str(round(np.std(scores1), 4)),
                        "No reweighting score: ", round(score1_max, 4),
                        "Scores1:", [round(i, 4) for i in scores1]],
                        subtitle="Clf trained on real/mc reweight, tested on real",
                        to_end=True)
#
#        ml_ana.data_ROC(reweight_mc_reweighted, reweight_mc_reweighted,
#                        curve_name="mc reweight all conv=5 vs mc reweight all",
#                        conv_ori_weights=5, conv_tar_weights=False, weights_ratio=1)
        #ml_ana.data_ROC(reweight_mc_reweighted, reweight_m
        #                curve_name="mc reweight vs mc reweight weights as events", conv_tar_weights=3)

        # compare weights_as_events vs normal weights
        reweight_mc_reweighted.plot(figure="weights as events vs normal weights",
                                    data_name="weights as events", weights_as_events=False)
        reweight_mc_reweighted.plot(figure="weights as events vs normal weights",
                                    data_name="normal weights", weights_as_events=False)
        reweight_real.plot(figure="weights as events vs normal weights", data_name="real data", weights_as_events=False)

        # normal KFold "how-well-distinguishable". Pay attention: Do not overfit your clf!
        ml_ana.data_ROC(reweight_mc_reweighted, reweight_real, classifier='xgb',
                        curve_name="mc reweighted vs real", n_folds=n_folds, conv_ori_weights=False, weights_ratio=1)
        ml_ana.data_ROC(reweight_mc, reweight_real, classifier='xgb',
                        curve_name="mc vs real (max)", n_folds=n_folds, conv_ori_weights=False, weights_ratio=1)
        reweight_real.plot(figure="real vs mc reweighted CV", title="Real data vs CV reweighted Monte-Carlo",
                           data_name="mc reweighted")
        reweight_mc_reweighted.plot(figure="real vs mc reweighted CV", data_name="real")


        out.save_fig(figure="New weights of total mc")
        plt.hist(reweight_mc_reweighted.get_weights(), bins=30)
    out.add_output("", subtitle="Cross validation reweight report", section="Precision scores of classification")
    score_list = [("GBReweighted: ", score_gb), ("mc as real (min): ", score_min), ("real as real (max): ", score_max)]
    for name, score in score_list:
        mean, std = round(np.mean(score), 4), round(np.std(score), 4)
        out.add_output(["Average score " + name + str(mean) + " +- " + str(std)])



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
    gb_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
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
    bins_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
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
