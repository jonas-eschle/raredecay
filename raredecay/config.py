# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:44:43 2016

The configuration file for external operations.

@author: Jonas Eschle "Mayou36"
"""
# Python 2 backwards compatibility overhead START
from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct,
                      open, pow, range, round, str, super, zip)
import sys
import warnings
import raredecay.meta_config

try:
    from future.builtins.disabled import (apply, cmp, coerce, execfile, file, long, raw_input,
                                          reduce, reload, unicode, xrange, StandardError)
    from future.standard_library import install_aliases

    install_aliases()
except ImportError as err:
    if sys.version_info[0] < 3:
        if raredecay.meta_config.SUPPRESS_FUTURE_IMPORT_ERROR:
            raredecay.meta_config.warning_occured()
            warnings.warn("Module future is not imported, error is suppressed. This means "
                          "Python 3 code is run under 2.7, which can cause unpredictable"
                          "errors. Best install the future package.", RuntimeWarning)
        else:
            raise err
    else:
        basestring = str

# Python 2 backwards compatibility overhead END

RUN_NAME = 'Classifier optimization'
run_message = str("This could be your advertisement" +
                  " ")

OUTPUT_CFG = dict(
        run_name=RUN_NAME,
        output_path=None,
        del_existing_folders=False,
        output_folders=dict(
                log="log",
                plots="plots",
                results="results",
                config="config"
                )
        )

save_fig_cfg = dict(
        file_format=['png', 'pdf'],
        to_pickle=True,
        dpi=150,
        figsize=(2, 10)
        )

# ==============================================================================
# LOGGER CONFIGURATION BEGIN
# ==============================================================================
logger_cfg = dict(
        logging_mode='both',  # define where the logger is written to
        # take 'both', 'file', 'console' or 'no'
        log_level_file='debug',
        # specifies the level to be logged to the file
        log_level_console='warning',  # 'warning',
        # specify the level to be logged to the console
        overwrite_file=True,
        # specifies whether it should overwrite the log file each time
        # or instead make a new one each run
        log_file_name='logfile_',
        # the beginning ofthe name of the logfile, like 'project1'
        log_file_dir=None  # will be set automatically
        )