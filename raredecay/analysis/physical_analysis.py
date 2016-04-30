# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:49:45 2016

@author: mayou

Contains the different run-modes for the machine-learning algorithms.
"""
from __future__ import division, absolute_import



import raredecay.meta_config
from raredecay import globals_
# debug
#import config as cfg


def run(runmode):
    """select the right runmode from the parameter and run it"""


    #run_config = 'config'

    _reweight1_comparison(runmode)
    #_simple_plot()
    globals_.finalize()  # finish the analysis, write output etc



def reweight(data_to_reweight, config_file=None):
    # specify the default configuration file. Can be changed.
    _DEFAULT_CONFIG_FILE = 'raredecay.run_config.reweight_cfg'
    # default: 'raredecay.run_config.reweight_cfg'

#PROTECTED, ALWAYS AT THE BEGINNING - PROTECTED, ALWAYS AT THE BEGINNING##
#########################################################################D
    if config_file is None:                                             #O
        raredecay.meta_config.run_config = _DEFAULT_CONFIG_FILE         #N
    import importlib                                                    #T
    cfg = importlib.import_module(raredecay.meta_config.run_config)     #
    # create logger                                                     #C
    from raredecay.tools import dev_tool                                #H
    logger = dev_tool.make_logger(__name__)                             #A
    logger.debug("config file used: " +                                 #N
                 str(raredecay.meta_config.run_config))                 #G
    globals_.initialize(**cfg)                                          #E
#########################################################################!
#PROTECTED, ALWAYS AT THE BEGINNING - PROTECTED, ALWAYS AT THE BEGINNING##

#==============================================================================
#     actual program start
#==============================================================================
    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_tools

    reweighter = ml_ana.reweight_mc_real(meta_cfg=cfg.reweight_meta_cfg,
                                         **cfg.reweight_cfg)
    # reweighter = ''  # load from pickle file
    new_weights = ml_ana.reweight_weights(data_to_reweight, reweighter)
    return data_tools.adv_return(new_weights)


def _simple_plot(config_file=None):
    # specify the default configuration file. Can be changed.
    _DEFAULT_CONFIG_FILE = 'raredecay.run_config.reweight1_comparison_cfg'
    # default 'run_config.reweight1_comparison_cfg'

#PROTECTED, ALWAYS AT THE BEGINNING - PROTECTED, ALWAYS AT THE BEGINNING##
#########################################################################D
    if config_file is None:                                             #O
        raredecay.meta_config.run_config = _DEFAULT_CONFIG_FILE         #N
    import importlib                                                    #T
    cfg = importlib.import_module(raredecay.meta_config.run_config)     #
    # create logger                                                     #C
    from raredecay.tools import dev_tool                                #H
    logger = dev_tool.make_logger(__name__)                             #A
    logger.debug("config file used: " +                                 #N
                 str(raredecay.meta_config.run_config))                 #G
    globals_.initialize(**cfg)                                          #E
#########################################################################!
#PROTECTED, ALWAYS AT THE BEGINNING - PROTECTED, ALWAYS AT THE BEGINNING##

#==============================================================================
#     actual program start
#==============================================================================

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
    # specify the default configuration file. Can be changed.
    _DEFAULT_CONFIG_FILE =     'raredecay.run_config.reweight1_comparison_cfg'
    # default: 'raredecay.run_config.reweight1_comparison_cfg'

#PROTECTED, ALWAYS AT THE BEGINNING - PROTECTED, ALWAYS AT THE BEGINNING##
#############################################################################D
    if config_file is None:                                                 #O
        raredecay.meta_config.run_config = _DEFAULT_CONFIG_FILE             #N
    import importlib                                                        #T
    cfg = importlib.import_module(raredecay.meta_config.run_config)         #
    # create logger                                                         #C
    from raredecay.tools import dev_tool                                    #H
    logger = dev_tool.make_logger(__name__)                                 #A
    logger.debug("config file used: " +                                     #N
                 str(raredecay.meta_config.run_config))                     #G
    globals_.initialize(logger_cfg=cfg.logger_cfg, **cfg.OUTPUT_CFG)        #!
##############################################################################
#PROTECTED, ALWAYS AT THE BEGINNING - PROTECTED, ALWAYS AT THE BEGINNING##

#==============================================================================
#     actual program start
#==============================================================================
    # TODO: remove import of matplotlib.pyplot after testing
    import matplotlib.pyplot as plt

    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools import data_storage

    # make data
    logger.info("Start with gb reweighter")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    # TODO: remove
    globals_.add_output(["hey na na na", "wie gehts so"])
    return None
    gb_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            #branches=['B_PT', 'nTracks', 'nSPDHits'
                                            #, 'h1_TRACK_TCHI2NDOF','B_ENDVERTEX_CHI2'
                                            #],
                                            reweighter='gb',
                                            meta_cfg=cfg.reweight_meta_cfg)
    #gb_reweighter = 'gb_reweighter1.pickle'
    ml_ana.reweight_weights(reweight_mc, #branches=['B_PT', 'nTracks', 'nSPDHits'
                                          #  , 'h1_TRACK_TCHI2NDOF','B_ENDVERTEX_CHI2'
                                           # ],
                            reweighter_trained=gb_reweighter)
    reweight_mc.plot2Dscatter('B_PT', 'nTracks', figure=2)
    reweight_real.plot2Dscatter('B_PT', 'nTracks', figure=2, color='r')
    gb_roc_auc = ml_ana.data_ROC(original_data=reweight_mc,
                                 target_data=reweight_real, curve_name="GB reweighted")
    reweight_mc.plot(figure="gradient boosted reweighting",
                     plots_name="comparison real-target", hist_settings={'bins':20})
    reweight_real.plot(figure="gradient boosted reweighting", hist_settings={'bins':20})
    plt.figure("Weights bg reweighter")
    plt.hist(reweight_mc.get_weights(), bins=20)
    plt.figure("Big weights (>4) bg reweighter")
    plt.hist([i for i in reweight_mc.get_weights() if i > 4], bins=200)
    print "mc weights sum", str(reweight_mc.get_weights().sum())
    print "real weights sum", str(reweight_real.get_weights().sum())
    plt.show()


    logger.info("Start with bins reweighter")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    logger.debug("plotted figure 2")
    bins_reweighter = ml_ana.reweight_mc_real(reweight_data_mc=reweight_mc,
                                            reweight_data_real=reweight_real,
                                            reweighter='bins',
                                            branches=['B_PT', 'nTracks', 'nSPDHits'
                                            #, 'h1_TRACK_TCHI2NDOF'
                                            ],
                                            meta_cfg=cfg.reweight_meta_cfg_bins)
    #bins_reweighter = 'bins_reweighter1.pickle'
    ml_ana.reweight_weights(reweight_mc, bins_reweighter, branches=['B_PT', 'nTracks', 'nSPDHits'
                                            #, 'h1_TRACK_TCHI2NDOF'
                                            ],)
    reweight_mc.plot(figure="binned reweighting",
                     plots_name="comparison real-target")
    reweight_real.plot(figure="binned reweighting")
    bins_roc_auc = ml_ana.data_ROC(original_data=reweight_mc,
                                   target_data=reweight_real, curve_name="Bins reweighted")
    # plt.show()


    logger.debug("starting with original")
    reweight_mc = data_storage.HEPDataStorage(**cfg.data.get('reweight_mc'))
    reweight_real = data_storage.HEPDataStorage(**cfg.data.get('reweight_real'))
    original_roc_auc = ml_ana.data_ROC(original_data=reweight_mc,
                                       target_data=reweight_real, curve_name="Original weights")
    reweight_mc.plot(figure="no reweighting",
                     plots_name="comparison real-target")
    reweight_real.plot(figure="no reweighting")



# temporary:
if __name__ == '__main__':
    run(1)
