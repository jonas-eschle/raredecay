# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:49:45 2016

@author: mayou

Contains the different run-modes for the machine-learning algorithms.
"""
import raredecay.meta_config

#import config as cfg


def run(runmode):
    """select the right runmode from the parameter and run it"""


    #run_config = 'config'

    _reweight1_comparison()



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


def _reweight1_comparison(config_file=None):
    # specifiy configuration file
    if config_file is None:
        raredecay.meta_config.run_config = 'raredecay.run_config.reweight1_comparison_cfg'  # 'run_config.reweight1_comparison_cfg'
    import importlib
    cfg = importlib.import_module(raredecay.meta_config.run_config)
    # create logger
    from raredecay.tools import dev_tool
    logger = dev_tool.make_logger(__name__)
    logger.debug("config file used: " +
                 str(raredecay.meta_config.run_config))
    # actual program start
    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_storage

    # make data
    logger.info("Start with gb reweighter")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real_no_sweights'))

    gb_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            reweighter='gb',
                                            meta_cfg=cfg.reweight_meta_cfg)
    #gb_reweighter = 'gb_reweighter1.pickle'
    ml_ana.reweight_weights(reweight_mc, gb_reweighter)
    gb_roc_auc = ml_ana.fast_ROC_AUC(original=reweight_mc, target=reweight_real)

    logger.info("Start with bins reweighter")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real_no_sweights'))

    bins_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            reweighter='bins',
                                            meta_cfg=cfg.reweight_meta_cfg_bins)
    #bins_reweighter = 'bins_reweighter1.pickle'
    ml_ana.reweight_weights(reweight_mc, bins_reweighter)



    bins_roc_auc = ml_ana.fast_ROC_AUC(original=reweight_mc,
                                        target=reweight_real)
    logger.debug("starting with original")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real_no_sweights'))
    original_roc_auc = ml_ana.fast_ROC_AUC(original=reweight_mc,
                                           target=reweight_real)
    print "original_roc_auc = ", original_roc_auc
    print "gb_roc_auc = ", gb_roc_auc
    print "bins_roc_auc = ", bins_roc_auc






def _test2():
    pass


def finalize():
    """Finalize the run: print to console and save output_string to file
    """
    print "function finalize not yet implemented"

# temporary:
if __name__ == '__main__':
    run(1)
