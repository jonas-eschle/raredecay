# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:32:17 2016

@author: Jonas Eschle "Mayou36"

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
    module-name changes, which can cause a crash of the program.
"""

# ==============================================================================
# DO NOT IMPORT ANY PACKAGE (run configuration) DEPENDENCY!
# ==============================================================================
from __future__ import division, absolute_import

import cPickle as pickle
import multiprocessing
import random


# ==============================================================================
# Parameters which can be changed WITHOUT affecting stability of a single run.
# Be aware: certain tasks like loading  a pickled file may fail if the file-
# endings are changed.
# ==============================================================================

# ------------------------------------------------------------------------------
# General run parameters
# ------------------------------------------------------------------------------

PROMPT_FOR_COMMENT = False  # let you add an extension to the run/file name
MULTITHREAD = True  # if False, no parallel work will be done
MULTIPROCESSING = True  # requires MULTITHREAD to be true, else it's False
n_cpu_max = 1  # VAGUE ESTIMATION but not a strict limit. If None, number of cores will be assigned
use_gpu = False  # If True, optimisation for GPU use is done (e.g. nn not parallel on cpu).
# This does NOT use the GPU yet, but "not use the cpu" where the GPU will be invoked
use_stratified_folding = True  # StratifiedKFolding is better, from a statistical point of view,
# but also needs more memory, mostly insignificantly but can be large


def get_n_cpu(n_cpu=None):
    """Return the number of cpus to use. None means all. Can be -1, -2..."""
    if n_cpu is None:
        n_cpu = -1
    if isinstance(n_cpu, int):
        if n_cpu < 0:
            n_cpu = max([n_cpu_max + n_cpu + 1, 1])  #
        n_cpu = min([n_cpu, n_cpu_max])
    return n_cpu


# set meta-config variables
def set_parallel_profile(n_cpu=-1, gpu_in_use=False, stratified_kfolding=True):
    """Set the number of cpus and whether a gpu is in use or not."""
    global MULTIPROCESSING, MULTITHREAD, n_cpu_max, use_gpu, use_stratified_folding
    use_stratified_folding = stratified_kfolding
    MULTIPROCESSING = MULTITHREAD = True
    if n_cpu == 1:
        n_cpu_max = 1
    elif n_cpu is None:
        pass
    elif isinstance(n_cpu, int):
        if n_cpu > 1:
            n_cpu_max = n_cpu
        elif n_cpu < 0:
            n_cpu_max = max([multiprocessing.cpu_count() + n_cpu + 1, 1])  # -1 is "all cpus"
        else:
            raise ValueError("Invalid n_cpu argument: " + str(n_cpu))
    else:
        raise TypeError("Wrong n_cpu argument, type: " + str(type(n_cpu)) + " not allowed")

    use_gpu = gpu_in_use if gpu_in_use is not None else use_gpu

# ------------------------------------------------------------------------------
#  Datatype ending variables
# ------------------------------------------------------------------------------

# The ending of a certain variable type. Change with caution and good reason.
PICKLE_DATATYPE = "pickle"  # default: 'pickle'
ROOT_DATATYPE = "root"  # default 'root'

# ------------------------------------------------------------------------------
# SHARED OBJECT PATHES INPUT & OUTPUT
# ------------------------------------------------------------------------------

# folder where the pickled objects are stored
PICKLE_PATH = '/home/mayou/Documents/uniphysik/Bachelor_thesis/analysis/pickle/'
# folder where the git-directory is located. Can be an empty string
GIT_DIR_PATH = "/home/mayou/Documents/uniphysik/Bachelor_thesis/" + \
               "python_workspace/raredecay/raredecay"

# ------------------------------------------------------------------------------
#  Debug related options
# ------------------------------------------------------------------------------

# This options should not directly affect the behaviour (except of speed etc)
# IF the right environment is used. Don't touch until you have good reasons to do.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # default: pickle.HIGHEST_PROTOCOL
SUPPRESS_WRONG_SKLEARN_VERSION = False  # Should NOT BE CHANGED.

# ==============================================================================
# Parameters which may affect stability
# setting for example MAX_AUTO_FOLDERS to 0, it will surely not work
# ==============================================================================
# ------------------------------------------------------------------------------
#  Limits for auto-methods
# ------------------------------------------------------------------------------

# If a folder already exists and no overwrite is in use, a new folder (with a
# trailing number) will be created. There can be set a limit to prevent a full
# disk in case of an endless loop-error or similar.
MAX_AUTO_FOLDERS = 10000  # max number of auto-generated folders by initialize
NO_PROMPT_ASSUME_YES = True  # no userinput required, assumes yes (e.g. when overwritting files)
MAX_ERROR_COUNT = 1000  # set a maximum number of possible errors (not able to save figure etc.)
# Criticals will end the run anyway.
MAX_FIGURES = 1000  # max number of figures to be plotted


# ==============================================================================
# DEFAULT SETTINGS for different things
# ==============================================================================

# ------------------------------------------------------------------------------
#  Output and plot configurations
# ------------------------------------------------------------------------------

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
    alpha=0.5,  # transparency [0.0, 1.0]
    histtype='stepfilled'
)

# Default configuration for most of the figures for save_fig from OutputHandler()
DEFAULT_SAVE_FIG = dict(
    file_format=['png', 'pdf'],  # default: ['png', 'svg'], the file formats
    dpi=150,                     # to be saved to. For implementations, see OutputHandler()
    to_pickle=True,  # whether to pickle the plot (and therefore be able to replot)
    # save_cfg=None
)

# Default configuration for additional figures (plots you mostly do not care
# about but may be happy to have them saved somewhere)
DEFAULT_EXT_SAVE_FIG = dict(
    file_format=['png', 'pdf'],
    to_pickle=True
    # save_cfg=None
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

# ------------------------------------------------------------------------------
#  Classifier configurations
# ------------------------------------------------------------------------------

# Some modules use classifiers for different tasks where it is mostly not
# important to have a fully optimized classifier but just a "good enough" one.
# Like in the data_ROC where you can see how well two datasets differ from
# each other.
# Changing this default values will surely affect your results (over- or
# underfitting for example), but is mostly not required at all.
DEFAULT_CLF_XGB = dict(
    n_estimators=150,  # default 75
    eta=0.1,  # default 0.1, learning-rate
    min_child_weight=0,  # #0 stage 2 to optimize
    max_depth=5,  # #6 stage 2 to optimize
    gamma=0.1,  # stage 3, minimum loss-reduction required to make a split.
    # Higher value-> more conservative
    subsample=0.8,  # stage 4, subsample of data. 1 means all data, 0.7 means only 70% of data
    # for a tree
    colsample=1
)

DEFAULT_CLF_TMVA = dict(
    method='kBDT'
)

DEFAULT_CLF_RDF = dict(
    n_estimators=150,
    max_features=None,
    # max_depth=100
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
    layers=[500, 500, 500],
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
    trainers=[{'optimize': 'adagrad', 'patience': 10, 'learning_rate': 0.1,
               'min_improvement': 0.01, 'momentum': 0.5, 'nesterov': True, 'loss': 'xe'}],
)

DEFAULT_CLF_KNN = dict(
    n_neigh=5
    )

# default clf config collection
DEFAULT_CLF_CONFIG = dict(
    xgb=DEFAULT_CLF_XGB,
    tmva=DEFAULT_CLF_TMVA,
    gb=DEFAULT_CLF_GB,
    ada=DEFAULT_CLF_ADA,
    nn=DEFAULT_CLF_NN,
    knn=DEFAULT_CLF_KNN,
    rdf=DEFAULT_CLF_RDF
)

# default clf names collection
DEFAULT_CLF_NAME = dict(
    xgb='XGBoost clf',
    tmva='TMVA clf',
    gb='Gradient Boosted Trees clf',
    ada='AdaBoost over Trees clf',
    nn='Theanets Neural Network clf',
    knn='K-Nearest Neighbour clf',
    rdf='Random Forest clf'
)
# ------------------------------------------------------------------------------
#  Hyper parameter optimization
# ------------------------------------------------------------------------------

# The backwards feature selection uses first all features and determines the ROC AUC.
# Then it removes one feature at a time, the one which yields the smallest difference
# to the 'all_features' roc auc is then removed. This continues until the smallest
# score difference is bigger then max_difference_feature_selection.
max_difference_feature_selection = 0.08  # the biggest score difference to 'all features'
# allowed in auc when removing features
DEFAULT_HYPER_GENERATOR = 'subgrid'  # The default cenerater for the hyperspace search

# ==============================================================================
# END OF CONFIGURABLE PARAMETERS - DO NOT CHANGE WHAT IS BELOW
# ==============================================================================

# DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE
# DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE
# DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE DO NOT CROSS THIS LINE


# ==============================================================================
# START INTERNAL CONFIGURATION - DO NOT CHANGE
# ==============================================================================

run_config = "raredecay.run_config.config"  # manipulated by OutputHandler()

loggers = {}

verbosity = 4
plot_verbosity = 3


def set_verbosity(new_verbosity):
    """Set the verbosity."""
    global verbosity
    verbosity = round(new_verbosity)
    _check_verbosity(verbosity)


def set_plot_verbosity(new_plot_verbosity):
    """Set the plot verbosity."""
    global plot_verbosity
    plot_verbosity = round(new_plot_verbosity)
    _check_verbosity(plot_verbosity)


def _check_verbosity(verbosity):
    if verbosity not in range(-1, 7):
        raise ValueError("Verbosity has to be int {0, 1, 2, 3, 4, 5}")

# ==============================================================================
# Random integer generator for pseudo random generator (or other things)
# ==============================================================================

rand_seed = random.randint(123, 1512412)  # 357422 or 566575
random.seed(rand_seed)


def randint():
    """Return random integer."""
    return random.randint(51, 523753)


def randfloat():
    """Return a random float between 0 and 1."""
    return random.random()


def set_seed(seed):
    """Set the global random seed."""
    global rand_seed
    rand_seed = seed
    random.seed(rand_seed)

# ------------------------------------------------------------------------------
# parallel profile
# ------------------------------------------------------------------------------

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

_error_count = 0  # increases if an error happens
_warning_count = 0  # increases if an error happens


def error_occured(max_error_count=MAX_ERROR_COUNT):
    """Call this function every time a non-critical error (saving etc) occurs."""
    global _error_count
    _error_count += 1
    if _error_count >= max_error_count:
        raise RuntimeError("Too many errors encountered from different sources")


def warning_occured():
    """Call this function every time a warning occurs."""
    global _warning_count
    _warning_count += 1

if __name__ == '__main__':
    pass
