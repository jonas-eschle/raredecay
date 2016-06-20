# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:32:17 2016

@author: mayou

| This module provides the meta-configuration.
| Mostly, you do not need to change this file.
| It contains:
 - (package-)global default variables for all modules
 - Debug-options which change some implementation on a basic level
 - Global configurations like the endings of specific files etc.

Variables:
---------
run_config:
    It provides the right config module depending on what was chosen
    in the run-methods.
    Should not be changed during the run, only once in the begining.
SUPPRESS_WRONG_SKLEARN_VERSION:
    This package was built for sklearn 0.17. With 0.18 there are some
    module-name changes, which can crash the program.
"""

#==============================================================================
# DO NOT IMPORT ANY PACKAGE (run configuration) DEPENDENCY!
#==============================================================================
from __future__ import division, absolute_import

import cPickle as pickle


#==============================================================================
# Parameters which can be changed WITHOUT affecting stability of a single run.
# Be aware: certain tasks like loading  a pickled file may fail if the file-
# endings are changed.
#==============================================================================

#------------------------------------------------------------------------------
# General run parameters
#------------------------------------------------------------------------------

PROMPT_FOR_COMMENT=True  # let you add a small extension to the run/file name and the run comment
MULTITHREAD = True  # if False, no parallel work will be done
MULTIPROCESSING = True  # requires MULTITHREAD to be true, else it's False
n_cpu_max = 2  # VAGUE ESTIMATION but not a strict limit. If None, number of cores will be assigned
use_gpu = True  # If True, optimisation for GPU use is done (e.g. nn not parallel on cpu)

#------------------------------------------------------------------------------
#  Datatype ending variables
#------------------------------------------------------------------------------


PICKLE_DATATYPE = "pickle"  # default: 'pickle'
ROOT_DATATYPE = "root"  # default 'root'

#------------------------------------------------------------------------------
# SHARED OBJECT PATHES INPUT & OUTPUT
#------------------------------------------------------------------------------

# folder where the pickled objects are stored
PICKLE_PATH = '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/pickle/'
GIT_DIR_PATH = "/home/mayou/Documents/uniphysik/Bachelor_thesis/python_workspace/HEP-decay-analysis/raredecay"

#------------------------------------------------------------------------------
#  Debug related options
#------------------------------------------------------------------------------

PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # default: pickle.HIGHEST_PROTOCOL
SUPPRESS_WRONG_SKLEARN_VERSION = False  # Should NOT BE CHANGED.

#==============================================================================
# Parameters which may affect stability
# setting for example MAX_AUTO_FOLDERS to 0, it will surely not work
#==============================================================================
#------------------------------------------------------------------------------
#  Limits for auto-methods
#------------------------------------------------------------------------------

MAX_AUTO_FOLDERS = 10000  # max number of auto-generated folders by initialize
NO_PROMPT_ASSUME_YES = False  # no userinput required, assumes yes (e.g. when overwritting files)
MAX_ERROR_COUNT = 1000  # set a maximum number of possible errors (like not able to save figure etc.)
MAX_FIGURES = 5000



#==============================================================================
# DEFAULT SETTINGS for different things
#==============================================================================

#------------------------------------------------------------------------------
#  Output and plot configurations
#------------------------------------------------------------------------------


# available output folders. Do NOT CHANGE THE KEYS as modules depend on them!
# You may add additional key-value pairs or just change some values
DEFAULT_OUTPUT_FOLDERS = dict(
    log="log",
    plots="plots",
    results="results",
    config="config"
)

DEFAULT_HIST_SETTINGS = dict(
    bins=40,
    normed=True,
    alpha=0.5  # transparency [0.0, 1.0]
)

DEFAULT_SAVE_FIG = dict(
    file_format=['png', 'svg'],
    to_pickle=True,
    plot=True,
    #save_cfg=None
)

DEFAULT_EXT_SAVE_FIG = dict(
    file_format=['png', 'svg'],
    to_pickle=True,
    plot=True,
    #save_cfg=None
)


DEFAULT_LOGGER_CFG = dict(
    logging_mode='console',   # define where the logger is written to
    # take 'both', 'file', 'console' or 'no'
    log_level_file='debug',  # 'debug', 'info', warning', 'error', 'critical'
    # specifies the level to be logged to the file
    log_level_console='debug',  # 'debug', 'info', warning', 'error', 'critical'
    # specify the level to be logged to the console
    overwrite_file=True,
    # specifies whether it should overwrite the log file each time
    # or instead make a new one each run
    log_file_name='AAlastRun',
    # the beginning ofthe name of the logfile, like 'project1'
    log_file_dir=DEFAULT_OUTPUT_FOLDERS.get('log')
)

#------------------------------------------------------------------------------
#  Classifier configurations
#------------------------------------------------------------------------------

DEFAULT_CLF_XGB = dict(
    n_estimators=10,
    eta=0.02,  # learning-rate
    max_depth=6,
    subsample=0.8
)

DEFAULT_CLF_TMVA = dict(
    method='kBDT'
)

DEFAULT_CLF_RDF = dict(
    n_estimators=200,
)

DEFAULT_CLF_GB = dict(
    n_estimators=200,
    learning_rate=0.15,
    max_depth=5,
    subsample=0.9,
    max_features=None
)

DEFAULT_CLF_ADA = dict(
    n_estimators=200,
    learning_rate=0.2
)

DEFAULT_CLF_KNN = dict(
    n_neigh = 5
)

#------------------------------------------------------------------------------
#  Hyper parameter optimization
#------------------------------------------------------------------------------

max_difference_feature_selection = 0.08  # the biggest difference to 'all features'
                                         # allowed in auc when removing features
DEFAULT_HYPER_GENERATOR = 'subgrid'

#==============================================================================
# END OF CONFIGURABLE PARAMETERS - DO NOT CHANGE WHAT IS BELOW
#==============================================================================




#==============================================================================
# START INTERNE CONFIGURATION - DO NOT CHANGE
#==============================================================================

run_config = None


#------------------------------------------------------------------------------
# parallel profile
#------------------------------------------------------------------------------

#==============================================================================
# ERROR HANDLING
#==============================================================================

_error_count = 0  # increases if an error happens
def error_occured(max_error_count=MAX_ERROR_COUNT):
    """Call this function every time a non-critical error (saving etc) occurs"""
    global _error_count
    _error_count += 1
    if _error_count >= max_error_count:
        raise RuntimeError("Too many errors encountered from different sources")

_warning_count = 0  # increases if an error happens
def warning_occured():
    """Call this function every time a warning occurs"""
    global _warning_count
    _warning_count += 1





if __name__ == '__main__':
    print "selftest of meta_config started"
    if DEFAULT_CLF_XGB.has_key(nthreads):
        raise ValueError("Do not specify threads. Use the parallel-profile")