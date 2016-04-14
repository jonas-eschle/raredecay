# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:49:45 2016

@author: mayou

Contains the different run-modes for the machine-learning algorithms.
"""
import raredecay.meta_config
# debug
#import config as cfg


def run(runmode):
    """select the right runmode from the parameter and run it"""


    #run_config = 'config'

    _reweight1_comparison(runmode)
    #_simple_plot()



def reweight(data_to_reweight, config_file=None):
    # specify configuration file
    if config_file is None:
        raredecay.meta_config.run_config = 'raredecay.run_config.reweight_cfg'
    import importlib
    cfg = importlib.import_module(raredecay.meta_config.run_config)
    # create logger
    from raredecay.tools import dev_tool
    logger = dev_tool.make_logger(__name__)
    logger.debug("config file used: " +
                 str(raredecay.meta_config.run_config))

    # actual program start
    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_tools

    reweighter = ml_ana.reweight_mc_real(meta_cfg=cfg.reweight_meta_cfg,
                                         **cfg.reweight_cfg)
    # reweighter = ''  # load from pickle file
    new_weights = ml_ana.reweight_weights(data_to_reweight, reweighter)
    return data_tools.adv_return(new_weights)


def _simple_plot(config_file=None):
    # specifiy configuration file
    if config_file is None:
        raredecay.meta_config.run_config = 'raredecay.run_config.reweight1_comparison_cfg'  # 'run_config.reweight1_comparison_cfg'
    import importlib
    cfg = importlib.import_module(raredecay.meta_config.run_config)
    # create logger
    from raredecay.tools import dev_tool
    logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)
    logger.debug("config file used: " +
                 str(raredecay.meta_config.run_config))

    # actual program start
    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_storage
    real_no_sweights = data_storage.HEPDataStorage(**cfg.data.get('reweight_real_no_sweights'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))



    reweight_real.plot(figure='sweights_vs_no_sweights')
    real_no_sweights.plot(figure='sweights_vs_no_sweights', plots_name='sweights versus no sweights')
    reweight_mc.plot(figure='mc_vs_real_no_sweights', plots_name='monte-carlo versus real no sweights')
    real_no_sweights.plot(figure='mc_vs_real_no_sweights', plots_name='monte-carlo versus real no sweights')


def _reweight1_comparison(i, config_file=None):
    # specifiy configuration file
    if config_file is None:
        raredecay.meta_config.run_config = 'raredecay.run_config.reweight1_comparison_cfg'  # 'run_config.reweight1_comparison_cfg'
    import importlib
    cfg = importlib.import_module(raredecay.meta_config.run_config)
    # create logger
    from raredecay.tools import dev_tool
    logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)
    logger.debug("config file used: " +
                 str(raredecay.meta_config.run_config))
    # actual program start
    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_storage

    # make data
    logger.info("Start with gb reweighter")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))

    gb_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            reweighter='gb',
                                            meta_cfg=cfg.reweight_meta_cfg)
    #gb_reweighter = 'gb_reweighter1.pickle'
    ml_ana.reweight_weights(reweight_mc, gb_reweighter)
    gb_roc_auc = ml_ana.fast_ROC_AUC(original=reweight_mc, target=reweight_real)
    reweight_mc.plot(figure="gradient boosted reweighting",
                     plots_name="comparison real-target")
    reweight_real.plot(figure="gradient boosted reweighting")

    logger.info("Start with bins reweighter")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))

    bins_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            reweighter='bins',
                                            meta_cfg=cfg.reweight_meta_cfg_bins)
    #bins_reweighter = 'bins_reweighter1.pickle'
    ml_ana.reweight_weights(reweight_mc, bins_reweighter)
    reweight_mc.plot(figure="binned reweighting",
                     plots_name="comparison real-target")
    reweight_real.plot(figure="binned reweighting")


    bins_roc_auc = ml_ana.fast_ROC_AUC(original=reweight_mc,
                                        target=reweight_real)
    logger.debug("starting with original")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    original_roc_auc = ml_ana.fast_ROC_AUC(original=reweight_mc,
                                           target=reweight_real)
    reweight_mc.plot(figure="no reweighting",
                     plots_name="comparison real-target")
    reweight_real.plot(figure="no reweighting")
    print "original_roc_auc = ", original_roc_auc[0], " = ", original_roc_auc[1]
    print "gb_roc_auc = ", gb_roc_auc[0], " = ", gb_roc_auc[1]
    print "bins_roc_auc = ", bins_roc_auc[0], " = ", bins_roc_auc[1]

    # temp plot
    import matplotlib.pyplot as plt
    orilabel="ROC curve original, AUC: " + str(round(original_roc_auc[0], 3)) + " or " + str(round(original_roc_auc[1], 3))
    gblabel="ROC curve gb, AUC: " + str(round(gb_roc_auc[0], 3)) + " or " + str(round(gb_roc_auc[1], 3))
    binslabel="ROC curve bin, AUC: " + str(round(bins_roc_auc[0], 3)) + " or " + str(round(bins_roc_auc[1], 3))
    columns=reweight_mc.get_labels(no_dict=True)
    plt.figure("roc auc different reweighter", figsize=(30, 40))
    plt.plot(original_roc_auc[2], original_roc_auc[3], label=orilabel)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of the branches: ' + str(columns))
    plt.legend(loc="lower right")

    plt.plot(gb_roc_auc[2], gb_roc_auc[3], label= gblabel )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.plot(bins_roc_auc[2], bins_roc_auc[3], label= binslabel )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower right")

    plt.savefig((str(i) + '-different_reweighters.png'), bbox_inches='tight')





def _test2():
    pass


def finalize():
    """Finalize the run: print to console and save output_string to file
    """
    print "function finalize not yet implemented"

# temporary:
if __name__ == '__main__':
    run(1)
