# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:45:51 2016

Contain methods to change settings in the whole package

@author: mayou
"""

from __future__ import division, absolute_import

from raredecay.run_config import config
from raredecay import meta_config


def parallel_profile(n_cpu=-1, gpu_in_use=False):
    """Set the parallel profile, the n_cpu to use

    Most of the functions do not have a number of threads option. You can
    change that here by either:

    - using a positive integer, which corresponds to the number of threads to
      use
    - using a negative integer. Then, -1 means to use all available (virtual)
      cores (resp. the number of threads). But sometimes you want to still work
      on your machine and have not all cpus used up. Therefore you can just use
      -2 (which will use all except of one; "all up to the second last"), -3
      (which will use all except two; "all up to the third last") etc.

    Parameters
    ----------
    n_cpu : int (not 0)
        The number of cpus to use, explanation see above.
    gpu_in_use : boolean
        If a gpu is in use (say for nn), you can set this boolean to true in
        order to avoid parallel computation of the nn or to activate any
        other implemented gpu assistance.


    """
    meta_config.set_parallel_profile(n_cpu=n_cpu, gpu_in_use=gpu_in_use)


def init_output_to_file(file_path, run_name="Test run", overwrite_existing=False,
                   run_message="This is a test-run to test the package"):
    """Saves output to file"""
    assert isinstance(file_path, str), "file_path has to be a string"
    assert isinstance(run_name, (str, int)), "run_name has to be a string or int"

    file_path = str(file_path) if isinstance(file_path, int) else file_path
    file_path += "" if file_path.endswith("/") else "/"

    config.RUN_NAME = str(run_name)
    config.run_message = str(run_message)
    config.OUTPUT_CFG['output_path'] = file_path
    config.OUTPUT_CFG['run_name'] = str(run_name)
    config.OUTPUT_CFG['del_existing_folders'] = overwrite_existing


def init_configure_logger(console_level='critical', file_level='debug'):
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


def init_user_input(prompt_for_input=True):
    """If called, you will be asked for input to name the specific run"""
    pass

