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

Bu2K1Jpsi_mc = dict(
    filenames=DATA_PATH+'original_data/Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='Bd2K1LL/DecayTree',
    branches=all_branches
)

cut_Bu2K1Jpsi_mc = dict(
    filenames=DATA_PATH+'CUT-Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='DecayTree',
    branches=all_branches

)

B2KpiLL_real = dict(
    filenames=DATA_PATH+'original_data/B2KpiLL-Collision12-MagDown-Stripping20r0p3.root',
    treename='Bd2K1LL/DecayTree',
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


# B-> K* reweighting
all_Kstar_branches = ['B0_PT',
                      'B0_ETA',
                      'B0_ENDVERTEX_CHI2',
                      'nSPDHits',
#                      'Kst_PT',
#                      'JPs_PT',
#                      'M1_PT',
#                      'M2_PT',
#                      'B0_IPCHI2_OWNPV',
#                      'B0_FDCHI2_OWNPV',
#                      'B0_DIRA_OWNPV'

                    ]
test_branches = [ 'B0_IPCHI2_OWNPV',
                 'B0FDCHI2_OWNPV',
                 'B0_DIRA_OWNPV'
                    ]

cut_Kstarmumu_mc = dict(
    filenames=DATA_PATH+'../Kstar/Bd2KstJPs_MM.root',
    treename='DecayTuple',
    branches=all_Kstar_branches
)

cut_sWeight_Kstarmumu_real = dict(
    filenames=DATA_PATH+'../Kstar/MM_LPT_sWeight.root',
    treename='DecayTree',
    branches=all_Kstar_branches
)
sWeights_Kstarmumu = dict(
    filenames=DATA_PATH+'../Kstar/MM_LPT_sWeight.root',
    treename='DecayTree',
    branches=['Cut_nsig_KstJPsMM_sw']
)

cut_Kstaree_mc = dict(
    filenames=DATA_PATH+'../Kstar/TODO',
    treename='TODO',
    branches=all_Kstar_branches
)


#------------------------------------------------------------------------------
# data in the HEPDataStorage-format (dicts containing all the parameters)
#------------------------------------------------------------------------------

#B2KpiLL_real_cut = dict(
#    data=cut_sWeight_B2KpiLL_real,
#    target=1,
#    data_name="B->KpiLL real data",
#    data_name_addition="cut",
#)

# gradient boosted reweighting
Bu2K1ee_mc_std = dict(
    data=Bu2K1ee_mc,
    sample_weights=None,
    data_name="Bu->K1ee monte-carlo"
)

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

# B -> K* reweighting

B2Kstarmumu_mc_cut = dict(
    data=cut_Kstarmumu_mc,
    sample_weights=None,
    data_name="B->K* MM monte-carlo",
    data_name_addition="cut"
)

B2Kstarmumu_real_cut_sweighted = dict(
    data=cut_sWeight_Kstarmumu_real,
    sample_weights=root2array(**sWeights_Kstarmumu),
    data_name="B->K* MM real",
    data_name_addition="cut & sweighted"
)

B2Kstaree_mc_std = dict(
    data="None so far",
    sample_weights=None,
    data_name="B->K* EE MC",
    data_name_addition="cut"
)

# Reweighting metric testing
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
#
#    reweight_mc=mc_testing,
#    reweight_real=real_testing,
#    reweight_apply=B2Kstaree_mc_std

# B -> K* configuration
#    reweight_mc=B2Kstarmumu_mc_cut,
#    reweight_real=B2Kstarmumu_real_cut_sweighted,
#    reweight_apply=B2Kstaree_mc_std


# B -> K1 configuration
    reweight_mc=B2K1Jpsi_mc_cut,
    reweight_real=B2KpiLL_real_cut_sweighted,
    reweight_apply=Bu2K1ee_mc_std,
    reweight2_mc=None,
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

reweight_cv_cfg = dict(
    n_folds=10,
    n_checks=10,
    plot_all=False,  # If True, all data (Folds and weights) are plotted. If False, only one example is
    #outdated: total_roc=True  # computes the ROC of all the reweighted samples. Only works if n_folds=n_checks
)

#------------------------------------------------------------------------------
# GENERAL REWEIGHTING PARAMETERS
#------------------------------------------------------------------------------

# branches to use for the reweighting

Kstar_reweight_branches = ['B0_PT',
                      'B0_ETA',
                      'B0_ENDVERTEX_CHI2',
                      'nSPDHits',
                      #'',

                    ]

K1_reweight_branches = ['B_PT', 'nTracks', 'nSPDHits',
                     'B_FDCHI2_OWNPV', 'B_DIRA_OWNPV'
                      #,'B_IPCHI2_OWNPV', 'l1_PT', 'l1_IPCHI2_OWNPV','B_ENDVERTEX_CHI2',
                      #'h1_IPCHI2_OWNPV', 'h1_PT', 'h1_TRACK_TCHI2NDOF'
              ]
reweight_branches = K1_reweight_branches  #['0', '1']  #Kstar_reweight_branches
reweight2_branches = None

# start configuration for gradient boosted reweighter

reweight_cfg = dict(
    reweighter='gb',  # can be a pickled reweighter or 'bins' resp 'gb'
    reweight_saveas='gb_reweighter1'  # if you want to save your reweighter
)
reweight_meta_cfg = dict(
    gb=dict(  # GB reweighter configuration
        n_estimators=20,  # 25
        max_depth=3,  # 6 or number of features
        learning_rate=0.1,  # 0.1
        min_samples_leaf=200,  # 200
        loss_regularization=7.0001,  #
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
    log_file_name='AAlastRun',
    # the beginning ofthe name of the logfile, like 'project1'
    log_file_dir=None  # will be set automatically
)


#==============================================================================
# LOGGER CONFIGURATION END
#==============================================================================



#==============================================================================
# SELFTEST BEGIN
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