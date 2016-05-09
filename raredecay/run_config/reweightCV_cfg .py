# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:26:13 2016

@author: mayou
"""
from __future__ import division, absolute_import

import cPickle as pickle
from root_numpy import root2array



# the name of the run and the output folder
RUN_NAME = 'test new data_ROC'
run_message = str("Test for new data_ROC implementation" +
                " ")
#==============================================================================
# PATHES BEGIN
#==============================================================================

#------------------------------------------------------------------------------
# SHARED OBJECT PATHES INPUT & OUTPUT
#------------------------------------------------------------------------------

# folder where the pickled objects are stored
PICKLE_PATH = '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/pickle/'

#------------------------------------------------------------------------------
# INPUT PATH
#------------------------------------------------------------------------------

#path where the data are stored  (folder)
DATA_PATH = '/home/mayou/Big_data/Uni/decay-data/B2KpiLL-Collision12-MagDown/'  # '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/data/'

#------------------------------------------------------------------------------
# OUTPUT PATHES
#------------------------------------------------------------------------------

# OUTPUT_PATH where the run will be stored
OUTPUT_CFG = dict(
    run_name=RUN_NAME,
    output_path='/home/mayou/Documents/uniphysik/Bachelor_thesis/output/',
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
Bu2K1ee_mc = dict(
    filenames=DATA_PATH+'DarkBoson/Bu2K1ee-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='Bd2K1LL/DecayTree',
    branches=['B_PT', 'nTracks']
)

Bu2K1Jpsi_mc = dict(
    filenames=DATA_PATH+'DarkBoson/Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='Bd2K1LL/DecayTree',
    branches=['B_PT', 'nTracks']
)
cut_Bu2K1Jpsi_mc = dict(
    filenames=DATA_PATH+'cut-data/CUT-Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='DecayTree',
    branches=['B_PT', 'nTracks', 'nSPDHits'
              , 'B_FDCHI2_OWNPV', 'B_DIRA_OWNPV'
              ,'B_IPCHI2_OWNPV', 'l1_PT', 'l1_IPCHI2_OWNPV','B_ENDVERTEX_CHI2',
              'h1_IPCHI2_OWNPV', 'h1_PT', 'h1_TRACK_TCHI2NDOF'
              ]#, 'B_TAU']

)
cut_B2KpiLL_real = dict(
    filenames=DATA_PATH+'cut-data/CUT-B2KpiLL-Collision12-MagDown-Stripping20r0p3.root',
    treename='DecayTree',
    branches=['B_PT', 'nTracks']
)
cut_sWeight_B2KpiLL_real = dict(
    filenames=DATA_PATH+'sweighted-data/B2KpiLL-Collision12-MagDown-Stripping20r0p3-Window-sWeights.root',
    treename='DecayTree',
    branches=['B_PT', 'nTracks', 'nSPDHits'
              , 'B_FDCHI2_OWNPV', 'B_DIRA_OWNPV'
              ,'B_IPCHI2_OWNPV', 'l1_PT', 'l1_IPCHI2_OWNPV','B_ENDVERTEX_CHI2',
              'h1_IPCHI2_OWNPV', 'h1_PT', 'h1_TRACK_TCHI2NDOF'
              ]#, 'B_TAU']

)
#------------------------------------------------------------------------------
# data in the HEPDataStorage-format (dicts containing all the parameters)
#------------------------------------------------------------------------------
B2KpiLL_real_cut = dict(
    data=cut_sWeight_B2KpiLL_real,
    target=1,
    data_name="B->KpiLL real data",
    data_name_addition="cut",
)

# gradient boosted reweighting
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

# train-test splitted data for reweighting
B2KpiLL_real_train = dict(
    data=dict(cut_sWeight_B2KpiLL_real, selection='B_M0234_Subst3_pi2p>4100'),
    sample_weights=root2array(**dict(cut_sWeight_B2KpiLL_real, branches=['signal_sw'], selection='B_M0234_Subst3_pi2p>4100')),
    data_name="B->KpiLL real data",
    data_name_addition="train set",
)
B2KpiLL_real_test = dict(
    data=dict(cut_sWeight_B2KpiLL_real, selection='B_M0234_Subst3_pi2p<4100'),
    sample_weights=root2array(**dict(cut_sWeight_B2KpiLL_real, branches=['signal_sw'], selection='B_M0234_Subst3_pi2p<4100')),
    data_name="B->KpiLL real data",
    data_name_addition="test set",
)
B2K1Jpsi_mc_train = dict(
    data=dict(cut_Bu2K1Jpsi_mc, selection='B_M0234_Subst3_pi2p>4200'),
    sample_weights=None,
    data_name="B->K1 J/Psi monte-carlo",
)
B2K1Jpsi_mc_test = dict(
    data=dict(cut_Bu2K1Jpsi_mc, selection='B_M0234_Subst3_pi2p<4200'),
    sample_weights=None,
    data_name="B->K1 J/Psi monte-carlo",
    data_name_addition="test set",
)


#------------------------------------------------------------------------------
# collection of all data
#------------------------------------------------------------------------------
# this dictionary will finaly be used in the code
data = dict(
    reweight_mc=B2K1Jpsi_mc_cut,
    reweight_real=B2KpiLL_real_cut_sweighted
)

#==============================================================================
# DATA END
#==============================================================================


#==============================================================================
# REWEIGHTING BEGIN
#==============================================================================

branches = ['B_PT', 'nTracks', 'nSPDHits']

# start configuration for gradient boosted reweighter
# TODO: remove the reweight_cfg?
reweight_cfg = dict(
    reweighter='gb',
    reweight_data_mc=cut_Bu2K1Jpsi_mc,
    reweight_data_real=cut_sWeight_B2KpiLL_real,
    reweight_saveas='gb_reweighter1'  # 'reweighter1.pickl'
)
reweight_meta_cfg = dict(
    gb=dict(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=100,  # 200
        loss_regularization=1000.0,  # 5.0
        gb_args=dict(
            subsample=0.8,
            #random_state=43,
            min_samples_split=40
        )
    ),
    bins=dict(
        n_bins=20,
        n_neighs=1
    )
).get(reweight_cfg.get('reweighter'))  # Don't change!



reweight_cfg_bins = dict(
    reweighter='bins',
    reweight_data_mc=cut_Bu2K1Jpsi_mc,
    reweight_data_real=cut_sWeight_B2KpiLL_real,
    reweight_saveas='bins_reweighter1'  # 'reweighter1.pickl'
)

reweight_meta_cfg_bins = dict(
    gb=dict(
        n_estimators=80
    ),
    bins=dict(
        n_bins=10,
        n_neighs=1
    )
).get(reweight_cfg_bins.get('reweighter'))  # Don't change!
# end config 1

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
    file_format=None,
    plot=True
)

save_ext_fig_cfg = dict(
    file_format=None,
    plot=False
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
