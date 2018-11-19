# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:44:43 2016

The configuration file for external operations.

@author: Jonas Eschle "Mayou36"
"""
# Python 2 backwards compatibility overhead START
from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct,  # noqa
                      open, pow, range, round, str, super, zip,
                      )  # noqa
import sys  # noqa
import warnings  # noqa
import raredecay.meta_config  # noqa

try:  # noqa
    from future.builtins.disabled import (apply, cmp, coerce, execfile, file, long, raw_input,  # noqa
                                      reduce, reload, unicode, xrange, StandardError,
                                      )  # noqa
    from future.standard_library import install_aliases  # noqa

    install_aliases()  # noqa
except ImportError as err:  # noqa
    if sys.version_info[0] < 3:  # noqa
        if raredecay.meta_config.SUPPRESS_FUTURE_IMPORT_ERROR:  # noqa
            raredecay.meta_config.warning_occured()  # noqa
            warnings.warn("Module future is not imported, error is suppressed. This means "  # noqa
                          "Python 3 code is run under 2.7, which can cause unpredictable"  # noqa
                          "errors. Best install the future package.", RuntimeWarning)  # noqa
        else:  # noqa
            raise err  # noqa
    else:  # noqa
        basestring = str  # noqa

# Python 2 backwards compatibility overhead END

__all__ = ['RUN_NAME', 'run_message', 'OUTPUT_CFG', 'save_fig_cfg', 'logger_cfg']

RUN_NAME = 'Default run name'
run_message = str("Default run message")

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
