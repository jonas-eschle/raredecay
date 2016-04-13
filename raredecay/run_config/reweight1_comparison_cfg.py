# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:26:13 2016

@author: mayou
"""
import cPickle as pickle
from root_numpy import root2array


# Datatype ending variables
PICKLE_DATATYPE = "pickle"  # default: 'pickle'
ROOT_DATATYPE = "root"  # default 'root'

# general variables
DATA_PATH = '/home/mayou/Big_data/Uni/decay-data/B2KpiLL-Collision12-MagDown'  # '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/data/'
PICKLE_PATH = '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/pickle/'
DATA_PATH += '/'

#DEBUG options
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # default: pickle.HIGHEST_PROTOCOL
FAST_CONVERSION = True  # default: True
MULTITHREAD = False  # not yet implemented


def path_test():
    for path in [DATA_PATH, PICKLE_PATH]:
        path += '/' if path[-1] not in ('/') else ""  # Don't change!

# reweighting

# root-dicts
Bu2K1ee_mc = dict(
    filenames=DATA_PATH+'DarkBoson/Bu2K1ee-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='Bd2K1LL/DecayTree',
    branches=["B_PT", "nTracks"]
)

Bu2K1Jpsi_mc = dict(
    filenames=DATA_PATH+'DarkBoson/Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='Bd2K1LL/DecayTree',
    branches=["B_PT", "nTracks"]
)
cut_Bu2K1Jpsi_mc = dict(
    filenames=DATA_PATH+'cut-data/CUT-Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
    treename='DecayTree',
    branches=["B_PT", "nTracks", 'Jpsi_P']#, 'B_TAU']
)
cut_B2KpiLL_real = dict(
    filenames=DATA_PATH+'cut-data/CUT-B2KpiLL-Collision12-MagDown-Stripping20r0p3.root',
    treename='DecayTree',
    branches=["B_PT", "nTracks"]
)
cut_sWeight_B2KpiLL_real = dict(
    filenames=DATA_PATH+'sweighted-data/B2KpiLL-Collision12-MagDown-Stripping20r0p3-Window-sWeights.root',
    treename='DecayTree',
    branches=["B_PT", "nTracks", 'Jpsi_P']#, 'B_TAU']
)

# data for HEPDataStorage
B2KpiLL_real_cut = dict(
    data=cut_sWeight_B2KpiLL_real,
    target=1,
    data_name="B->KpiLL real data",
    data_name_addition="cut",
)
B2KpiLL_real_cut_sweighted = dict(
    data=cut_sWeight_B2KpiLL_real,
    target=1,
    sample_weights=root2array(**dict(cut_sWeight_B2KpiLL_real, branches=['signal_sw'])),
    data_name="B->KpiLL real data",
    data_name_addition="cut & sweighted",
)
B2K1Jpsi_mc_cut = dict(
    data=cut_Bu2K1Jpsi_mc,
    target=0,
    sample_weights=None,
    data_name="B->K1 J/Psi monte-carlo",
    data_name_addition="cut"
)
# collection of all data
data = dict(
        reweight_mc=B2K1Jpsi_mc_cut,
        reweight_real_no_sweights=B2KpiLL_real_cut,
        reweight_real=B2KpiLL_real_cut_sweighted
)


# start default config
reweight_cfg = dict(
    reweighter='gb',
    reweight_data_mc=cut_Bu2K1Jpsi_mc,
    reweight_data_real=cut_sWeight_B2KpiLL_real,
    reweight_saveas='gb_reweighter1'  # 'reweighter1.pickl'
)
reweight_meta_cfg = dict(
    gb=dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03
    ),
    bins=dict(
        n_bins=20
    )
).get(reweight_cfg.get('reweighter'))  # Don't change!
# end default config

# start config 1
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
        n_bins=15,
        n_neighs=3
    )
).get(reweight_cfg_bins.get('reweighter'))  # Don't change!
# end config 1


# draw configuration
hist_cfg_std = dict(
    bins=40,
    normed=True,
    alpha=0.5  # transparency [0.0, 1.0]
)







pathes_to_add = []

# configure LOGGER
# -----------------------------------------------------------
logger_cfg = dict(
    logging_mode='both',   # define where the logger is written to
    # take 'both', 'file', 'console' or 'no'
    log_level_file='debug',
    # specifies the level to be logged to the file
    log_level_console='debug',
    # specify the level to be logged to the console
    overwrite_file=True,
    # specifies whether it should overwrite the log file each time
    # or instead make a new one each run
    log_file_name='AAlastRun',
    # the beginning ofthe name of the logfile, like 'project1'
    log_file_dir='../log'
)




def _selftest_system():
    """Test the configuration regarding the system-relevant parameters"""

    # test pathes_to_add
    if not all(type(i) == str for i in pathes_to_add):
        raise TypeError(str(filter(lambda i: type(i) != str, pathes_to_add)) +
                        " not of type string")
    # test logging_mode
    if logger_cfg['logging_mode'] not in ("both", "file", "console"):
        raise ValueError(str(logger_cfg['logging_mode']) +
                         ": invalid choice for logging_mode")

    # test loggerLevel


    # test logfile directory

def test_all():
    _selftest_system()

if __name__ == "__main__":
    test_all()
    print "config file succesfully tested!"
