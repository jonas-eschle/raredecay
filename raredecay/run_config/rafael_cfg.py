# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:26:13 2016

@author: mayou
"""
from __future__ import division, absolute_import

from root_numpy import root2array



# the name of the run and the output folder
RUN_NAME = 'TODO runname'
run_message = str("testing TODO" +
                "with TODO runmessage ")
#==============================================================================
# PATHES BEGIN
#==============================================================================



#------------------------------------------------------------------------------
# INPUT PATH
#------------------------------------------------------------------------------

# TODO
#path where the data are stored  (folder)
DATA_PATH = '/home/mayou/Big_data/Uni/decay-data/reweighting/'  # '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/data/'

#------------------------------------------------------------------------------
# OUTPUT PATHES
#------------------------------------------------------------------------------

# OUTPUT_PATH where the run will be stored
OUTPUT_CFG = dict(
    run_name=RUN_NAME,
    output_path='/home/mayou/Documents/uniphysik/Bachelor_thesis/output/',  # TODO
    del_existing_folders=False,
    output_folders=dict(
        log="log",
        plots="plots",
        results="results",
        config="config"
    )
)
#==============================================================================
# PATHES END
#==============================================================================



# REWEGIHTING

#==============================================================================
# DATA BEGIN
#==============================================================================

#------------------------------------------------------------------------------
# root data (dictionaries with parameters for root2array)
#------------------------------------------------------------------------------

# probably TODO, adjust for your data
# you can specify any branches you like here. To use only several ones for the
# reweighting/clf training, it is best to specify it in the method you call, it
# should mostly be possible to provide it as an argument.
all_branches = ['B_PT', 'nTracks', 'nSPDHits'
              , 'B_FDCHI2_OWNPV', 'B_DIRA_OWNPV'
              ,'B_IPCHI2_OWNPV', 'l1_PT', 'l1_IPCHI2_OWNPV','B_ENDVERTEX_CHI2',
              'h1_IPCHI2_OWNPV', 'h1_PT', 'h1_TRACK_TCHI2NDOF'
              ]



Bu2K1ee_mc = dict(
    filenames=DATA_PATH+'Bu2K1ee-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='Bd2K1LL/DecayTree',
    branches=all_branches
)


cut_Bu2K1Jpsi_mc = dict(
    filenames=DATA_PATH+'CUT-Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='DecayTree',
    branches=all_branches

)

cut_B2KpiLL_real = dict(
    filenames=DATA_PATH+'CUT-B2KpiLL-Collision12-MagDown-Stripping20r0p3.root',
    treename='Bd2K1LL/DecayTree',
    branches=all_branches
)

cut_sWeight_B2KpiLL_real = dict(
    filenames=DATA_PATH+'B2KpiLL-Collision12-MagDown-Stripping20r0p3-Window-sWeights.root',
    treename='DecayTree',
    branches=all_branches

)


#------------------------------------------------------------------------------
# data in the HEPDataStorage-format (dicts containing all the parameters)
#------------------------------------------------------------------------------


# gradient boosted reweighting
# good example with weights added from the rood file
B2KpiLL_real_cut_sweighted = dict(
    data=cut_sWeight_B2KpiLL_real,
    sample_weights=root2array(**dict(cut_sWeight_B2KpiLL_real, branches=['signal_sw'])),
    data_name="B->KpiLL real data",
    data_name_addition="cut & sweighted",
)
B2K1Jpsi_mc_cut = dict(
    data=cut_Bu2K1Jpsi_mc,
    sample_weights=None,
    data_name="B->K1 J/Psi monte-carlo",
    data_name_addition="cut"
)


# Reweighting metric testing, interesting to test functions by using artificial
# distributions
import pandas as pd
import numpy as np
testing_size = 15000
mc_testing = dict(
    data=pd.DataFrame({'0': np.random.normal(size=testing_size),
                       '1': np.random.normal(size=testing_size),
                       '2': np.random.normal(size=testing_size),
                       '3': np.random.normal(size=testing_size)}),
    sample_weights=None,
    data_name="mc gaussian dist",
    data_name_addition=""
)
real_testing = dict(
    data=pd.DataFrame({'0': np.random.normal(loc=0.5, scale=1.0, size=testing_size),
                       '1': np.random.normal(loc=0.5, scale=1.0, size=testing_size),
                       '2': np.random.normal(loc=0.5, scale=1.0, size=testing_size),
                       '3': np.random.normal(loc=0.5, scale=1.0, size=testing_size)}),
    sample_weights=None,
    data_name="real gaussian dist",
    data_name_addition=""
)

#------------------------------------------------------------------------------
# collection of all data
#------------------------------------------------------------------------------
# TODO: select which data to use
# this dictionary will finally be used in the code
data = dict(


# B -> K1 configuration
    reweight_mc=B2K1Jpsi_mc_cut,
    reweight_real=B2KpiLL_real_cut_sweighted,
    reweight_apply=None,  # 
    reweight2_mc=None,  # not used, just as an example
    reweight2_real=None
)

#==============================================================================
# DATA END
#==============================================================================


#==============================================================================
# REWEIGHTING BEGIN
#==============================================================================

#------------------------------------------------------------------------------
# ONLY FOR CV-RUN BEGIN
#------------------------------------------------------------------------------
# this is the place to change the scripts behaviour
reweight_cv_cfg = dict(
    n_folds=10,
    n_checks=10,
    make_plot=True,  # True: makes plot, False: makes no plots, 'all': plots all folds, not only examples
    total_roc=True  # computes the ROC of all the reweighted samples. Default is True.
)

#------------------------------------------------------------------------------
# GENERAL REWEIGHTING PARAMETERS
#------------------------------------------------------------------------------

# branches to use for the reweighting training

K1_reweight_branches = ['B_PT',
                        'nTracks',
                        'nSPDHits',
                        'B_FDCHI2_OWNPV',
                        'B_DIRA_OWNPV'
                      #,'B_IPCHI2_OWNPV', 'l1_PT', 'l1_IPCHI2_OWNPV','B_ENDVERTEX_CHI2',
                      #'h1_IPCHI2_OWNPV', 'h1_PT', 'h1_TRACK_TCHI2NDOF'
              ]
reweight_branches = K1_reweight_branches  
reweight2_branches = None  # just for demo

# start configuration for gradient boosted reweighter
# this are the parameters for the reweighting.

reweight_cfg = dict(
    reweighter='gb',  # can be a pickled reweighter or 'bins' resp 'gb'. You may change it to bins for testing
    reweight_saveas='gb_reweighter1'  # if you want to save your reweighter, otherwise None
)

# This is the place to change the reweighter configuration
reweight_meta_cfg = dict(
    gb=dict(  # GB reweighter configuration
        n_estimators=20,  # 25
        max_depth=3,  # 6 or number of features
        learning_rate=0.1,  # 0.1
        min_samples_leaf=200,  # 200
        loss_regularization=7.0,  #
        gb_args=dict(
            subsample=0.6, # 0.8
            #random_state=43,
            min_samples_split=200  # 200

        )
    ),
    bins=dict(  # Bins reweighter configuration
        n_bins=15,
        n_neighs=1
    )
).get(reweight_cfg.get('reweighter', None))  # Don't change!


#==============================================================================
# REWEIGHTER END
#==============================================================================


#==============================================================================
# PLOT CONFIGURATIONS BEGIN
#==============================================================================

# hist cfg mostly not usable. Others should have effect, but not mandatory
hist_cfg_std = dict(
    bins=40,
    normed=True,
    alpha=0.5  # transparency [0.0, 1.0]
)

save_fig_cfg = dict(
    file_format=['png', 'svg'],
    plot=True
)

save_ext_fig_cfg = dict(
    file_format=['png', 'svg'],
    plot=True
)

#==============================================================================
# PLOT CONFIGURATIONS END
#==============================================================================


#==============================================================================
# LOGGER CONFIGURATION BEGIN
#==============================================================================

# the logger configuration. Feel free to change the things.
logger_cfg = dict(
    logging_mode='both',   # define where the logger is written to
    # take 'both', 'file', 'console' or 'no'
    log_level_file='debug',
    # specifies the level to be logged to the file
    log_level_console='debug', #'warning',
    # specify the level to be logged to the console
    overwrite_file=True,
    # specifies whether it should overwrite the log file each time
    # or instead make a new one each run
    log_file_name='Logfile',
    # the beginning ofthe name of the logfile, like 'project1'
    log_file_dir=None  # will be set automatically
)


#==============================================================================
# LOGGER CONFIGURATION END
#==============================================================================



#==============================================================================
# SELFTEST BEGIN, OUTDATED! 
#==============================================================================
def _selftest_system():
    """Test the configuration regarding the system-relevant parameters"""

    def path_test():
        for path in [DATA_PATH, PICKLE_PATH]:
            path += '/' if path[-1] not in ('/') else ""  # Don't change!


    # test logging_mode
    if logger_cfg['logging_mode'] not in ('both', 'file', 'console'):
        raise ValueError(str(logger_cfg['logging_mode']) +
                         ": invalid choice for logging_mode")

    # test loggerLevel


    # test logfile directory

def test_all():
    _selftest_system()

if __name__ == '__main__':
    test_all()
    print "config file succesfully tested!"
