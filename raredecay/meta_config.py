# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:32:17 2016

@author: mayou

| This module provides the meta-configuration.
| Mostly, you do not need to change this file or only some small parts of it.
| Things you may want to change include whether to
    - promt for a addition to the file name
    - number of cores to use
    - path for pickle-files
    - default configuration for classifiers, figure saving and more
| It contains furthermore:
 - (package-)global default variables for all modules.
 - Debug-options which change some implementation on a basic level like protocols.
 - Global configurations like the endings of specific file-types etc.

The explanation to the variables is available as a comment behind each.

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

PROMPT_FOR_COMMENT=True  # let you add an extension to the run/file name
MULTITHREAD = True  # if False, no parallel work will be done
MULTIPROCESSING = True  # requires MULTITHREAD to be true, else it's False
n_cpu_max = 6  # VAGUE ESTIMATION but not a strict limit. If None, number of cores will be assigned
use_gpu = False  # If True, optimisation for GPU use is done (e.g. nn not parallel on cpu).
                # This does NOT use the GPU yet, but "not use the cpu" where the GPU will be invoked

#------------------------------------------------------------------------------
#  Datatype ending variables
#------------------------------------------------------------------------------

# The ending of a certain variable type. Change with caution and good reason.
PICKLE_DATATYPE = "pickle"  # default: 'pickle'
ROOT_DATATYPE = "root"  # default 'root'

#------------------------------------------------------------------------------
# SHARED OBJECT PATHES INPUT & OUTPUT
#------------------------------------------------------------------------------

# folder where the pickled objects are stored
PICKLE_PATH = '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/pickle/'
# folder where the git-directory is located. Can be an empty string
GIT_DIR_PATH = "/home/mayou/Documents/uniphysik/Bachelor_thesis/python_workspace/HEP-decay-analysis/raredecay"

#------------------------------------------------------------------------------
#  Debug related options
#------------------------------------------------------------------------------

# This options should not directly affect the behaviour (except of speed etc)
# IF the right environment is used. Don't touch until you have good reasons to do.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # default: pickle.HIGHEST_PROTOCOL
SUPPRESS_WRONG_SKLEARN_VERSION = False  # Should NOT BE CHANGED.

#==============================================================================
# Parameters which may affect stability
# setting for example MAX_AUTO_FOLDERS to 0, it will surely not work
#==============================================================================
#------------------------------------------------------------------------------
#  Limits for auto-methods
#------------------------------------------------------------------------------

# If a folder already exists and no overwrite is in use, a new folder (with a
# trailing number) will be created. There can be set a limit to prevent a full
# disk in case of an endless loop-error or similar.
MAX_AUTO_FOLDERS = 10000  # max number of auto-generated folders by initialize
NO_PROMPT_ASSUME_YES = False  # no userinput required, assumes yes (e.g. when overwritting files)
MAX_ERROR_COUNT = 1000  # set a maximum number of possible errors (like not able to save figure etc.)
                        # Criticals will end the run anyway.
MAX_FIGURES = 1000  # max number of figures to be plotted

#==============================================================================
# DEFAULT SETTINGS for different things
#==============================================================================

#------------------------------------------------------------------------------
#  Output and plot configurations
#------------------------------------------------------------------------------

# available output folders. Do NOT CHANGE THE KEYS as modules depend on them!
# You may add additional key-value pairs or just change some values

# The name of the folders created inside the run-folder
DEFAULT_OUTPUT_FOLDERS = dict(
    log="log",  # contains the logger informations
    plots="plots",  # contains all the plots
    results="results",  # contains the written output
    config="config"  # NOT YET IMPLEMENTED, but cound contain the config file used
)

# The default histogram settings used for some plots
DEFAULT_HIST_SETTINGS = dict(
    bins=40,  # default: 40
    normed=True,  # default: True, useful for shape comparison of distributions
    alpha=0.5  # transparency [0.0, 1.0]
)

# Default configuration for most of the figures for save_fig from OutputHandler()
DEFAULT_SAVE_FIG = dict(
    file_format=['png', 'svg'],  # default: ['png', 'svg'], the file formats
                                 # to be saved to. For implementations, see OutputHandler()
    to_pickle=True,  # whether to pickle the plot (and therefore be able to replot)
    plot=True,  # whether to plot the figure. If False, the figure will only be saved
    #save_cfg=None
)

# Default configuration for additional figures (plots you mostly do not care
# about but may be happy to have them saved somewhere)
DEFAULT_EXT_SAVE_FIG = dict(
    file_format=['png', 'svg'],
    to_pickle=True,
    plot=True,
    #save_cfg=None
)

# A logger writes some stuff during the run just for the control of the
# correct execution. The log will be written to console, to file, or both.
# Each message has a level ranging from the lowest (most unimportant) 'debug'
# to 'critical'. You can specify which level (+ the more important one) will
# appear where.
# Example: you can set console to 'error'. and file to 'info'. This way you
# collect also seemingly unneccesary informations (which are maybe later nice
# to check if a variable was meaningful) but on the screen you will only see
# if an error or critical occurs.
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

# Some modules use classifiers for different tasks where it is mostly not
# important to have a fully optimized classifier but just a "good enough" one.
# Like in the data_ROC where you can see how well two datasets differ from
# each other.
# Changing this default values will surely affect your results (over- or
# underfitting for example), but is mostly not required at all.
DEFAULT_CLF_XGB = dict(
    n_estimators=75,  # default 75
    eta=0.1,  # default 0.1, learning-rate
    min_child_weight=8,  # #0 stage 2 to optimize
    max_depth=3,  # #6 stage 2 to optimize
    gamma=4.6,  # stage 3, minimum loss-reduction required to make a split. Higher value-> more conservative
    subsample=0.8, # stage 4, subsample of data. 1 means all data, 0.7 means only 70% of data for a tree
    colsample=1
)

DEFAULT_CLF_TMVA = dict(
    method='kBDT'
)

DEFAULT_CLF_RDF = dict(
    n_estimators=150,
    max_features=None,
    #max_depth=100
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

DEFAULT_CLF_NN = dict(
    layers=[300, 100],
    hidden_activation='logistic',
    output_activation='linear',
    input_noise=0,  # [0,1,2,3,4,5,10,20],
    hidden_noise=0,
    input_dropout=0,
    hidden_dropout=0.03,
    decode_from=1,
    weight_l1=0.01,
    weight_l2=0.01,
    scaler='standard',
    trainers=[{'optimize': 'adagrad', 'patience': 15, 'learning_rate': 0.1, 'min_improvement': 0.01,
               'momentum':0.5, 'nesterov':True, 'loss': 'xe'}],
)

DEFAULT_CLF_KNN = dict(
    n_neigh = 5
)

# default clf config collection
DEFAULT_CLF_CONFIG = dict(
    xgb=DEFAULT_CLF_XGB,
    tmva=DEFAULT_CLF_TMVA,
    gb=DEFAULT_CLF_GB,
    ada=DEFAULT_CLF_ADA,
    nn=DEFAULT_CLF_NN,
    knn=DEFAULT_CLF_KNN,
    nn=DEFAULT_CLF_NN
)

# default clf names collection
DEFAULT_CLF_NAME = dict(
    xgb='XGBoost clf',
    tmva='TMVA clf',
    gb='Gradient Boosted Trees clf',
    ada='AdaBoost over Trees clf',
    nn='Theanets Neural Network clf',
    knn='K-Nearest Neighbour clf'
)
#------------------------------------------------------------------------------
#  Hyper parameter optimization
#------------------------------------------------------------------------------

# The backwards feature selection uses first all features and determines the ROC AUC.
# Then it removes one feature at a time, the one which yields the smallest difference
# to the 'all_features' roc auc is then removed. This continues until the smallest
# score difference is bigger then max_difference_feature_selection.
max_difference_feature_selection = 0.08  # the biggest score difference to 'all features'
                                         # allowed in auc when removing features
DEFAULT_HYPER_GENERATOR = 'subgrid'  # The default cenerater for the hyperspace search

#==============================================================================
# END OF CONFIGURABLE PARAMETERS - DO NOT CHANGE WHAT IS BELOW
#==============================================================================

# DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE
# DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE
# DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE


#==============================================================================
# START INTERNAL CONFIGURATION - DO NOT CHANGE
#==============================================================================

run_config = None  # manipulated by OutputHandler()

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
    pass