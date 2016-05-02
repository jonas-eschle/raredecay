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


run_config = None  # 'config'




#==============================================================================
#  Datatype ending variables
#==============================================================================
PICKLE_DATATYPE = "pickle"  # default: 'pickle'
ROOT_DATATYPE = "root"  # default 'root'

#==============================================================================
#  DEBUG options
#==============================================================================
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # default: pickle.HIGHEST_PROTOCOL
MULTITHREAD = False  # not yet implemented
SUPPRESS_WRONG_SKLEARN_VERSION = False  # Should NOT BE CHANGED.
MAX_AUTO_FOLDERS = 10000  # max number of auto-generated folders by initialize
NO_PROMPT_ASSUME_YES = False  # no userinput required, assumes yes (e.g. when overwritting files)
MAX_ERROR_COUNT = 10000  # set a maximum number of possible errors (like not able to save figure etc.)


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

#==============================================================================
# DEFAULT SETTINGS
#==============================================================================

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


if __name__ == '__main__':
    pass