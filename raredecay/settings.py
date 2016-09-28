# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:45:51 2016

Contain methods to change settings in the whole package

@author: mayou
"""

from __future__ import division, absolute_import

from raredecay.meta_config import loggers
from raredecay.run_config import config




def output_to_file(file_path, overwrite_existing=False):
    """Saves output to file"""
    assert isinstance(file_path, str), "file_path has to be a string"


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

def configure_logger(console_level='critical', file_level='debug'):
    """Call before imports! Set the logger-level

    The package contains several loggers which will print/save to file some
    information. What kind of information is handled can be changed by this
    function. The higher (severer) the level, the less will be displayed.

    Parameters
    ----------
    console_level : None or str {'debug', 'info', 'warning', 'error', 'critical'}
        Define the logging-level of the console handler. None means no output
        at all.
    file_level : None or str {'debug', 'info', 'warning', 'error', 'critical'}
        Define the logging-level of the file handler. None means no output
        at all.
    """


    if console_level is None and file_level is None:
        logging_mode = None
    elif console_level is None:
        logging_mode = 'file'
    elif file_level is None:
        logging_mode = 'console'
    else:
        logging_mode = 'both'

    for level in (console_level, file_level):
        assert level in (None, 'debug', 'info', 'warning', 'error', 'critical'), "invalid logger level"

    config.logger_cfg['logging_mode'] = logging_mode
    config.logger_cfg['log_level_file'] = file_level
    config.logger_cfg['log_level_console'] = console_level
    config.logger_cfg['overwrite_file'] = True
    config.logger_cfg['log_file_name'] = 'logfile_'


def change_logger(logger_level='info', change_mode='both'):
    """Change the behaviour of the loggers"""
    pass



def user_input(ask_for_input):
    """If True, it will ask for user input (at the beginning and the end)"""
    pass

