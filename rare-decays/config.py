# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:26:13 2016

@author: mayou
"""
import cPickle as pickle

PICKLE_DATATYPE = "pickle"
ROOT_DATATYPE = "root"

# general variables
DATA_PATH = '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/data/'
PICKLE_PATH = '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/pickle/'

PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def path_test():
    for path in [DATA_PATH, PICKLE_PATH]:
        path += '/' if path[-1] not in ('/') else ""  # Don't change!

# reweighting
reweight_cfg = dict(
    reweighter='gb',
    reweight_data_mc=dict(
        filenames=DATA_PATH+'DarkBoson/Bu2K1ee-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
        treename='Bd2K1LL/DecayTree',
        branches=["B_PT", "nTracks"]
    ),
    reweight_data_real=dict(
        filenames=DATA_PATH+'DarkBoson/Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root',
        treename='Bd2K1LL/DecayTree',
        branches=["B_PT", "nTracks"]

    ),
    reweight_saveas=None  # 'reweighter1.pickl'
)

reweight_meta_cfg = dict(
    gb=dict(
        n_estimators=50
    ),
    bins=dict(
        n_bins=20
    )
).get(reweight_cfg.get('reweighter'))  # Don't change!










pathes_to_add = []

# configure LOGGER
# -----------------------------------------------------------
logger_cfg = dict(
    logging_mode='both',   # define where the logger is written to
    # take 'both', 'file', 'console' or 'no'
    log_level_file='debug',
    # specifies the level to be logged to the file
    log_level_console='debug',
    # specifies the level to be logged to the console
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
