# -*- coding: utf-8 -*-
"""

@author: Jonas Eschle "Mayou36"


DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat

DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!


Contains several useful tools for all kind of programs
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
    from past.builtins import basestring  # noqa
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

import pandas as pd
import numpy as np
import collections

import raredecay.meta_config as meta_cfg


def make_logger(module_name, logging_mode='both', log_level_file='debug',
                log_level_console='debug', overwrite_file=True,
                log_file_name='AAlast_run', log_file_dir=None):
    """Return a logger with a console-/filehandler or both.

    A useful tool to log the run of the program and debug or control it. With
    logger.debug("message") a loging message is produced consisting of:
    timestamp(from year to ms) - module_name - logger level - message
    This can be either written to files, the console or both.

    Parameters
    ----------
    module_name : string
        Name of the logger, shown in output. Best choose __name__
    logging_mode : {'both', 'file', 'console'}
        Which logger handler is used; where the log is printed to.
    log_level_file : {'debug','info','warning','error','critical'}
        Which level of messages are logged. A lower level (left) always also
        includes the higher (right) levels, but not the other way around.
        This level specifies the level for the file log (if enabled).
    log_level_console : {'debug','info','warning','error','critical'}
        Level for console log (if enabled). See also log_level_file.
    overwrite_file : boolean
        If enabled, the logfiled gets overwritten at every run.
        Otherwise, a new logfile is created.
    log_file_name : string
        The name of the logfile
    log_file_dir : string
        The directory to save the logfile.
    Returns
    -------
    out : loggerObject
        Logger instance

    Examples
    --------
    >>> my_logger = make_logger(__name__)
    >>> my_logger.info("hello world")
    """
    import logging
    from time import strftime

    module_name = entries_to_str(module_name)
    logging_mode = entries_to_str(logging_mode)
    log_level_file = entries_to_str(log_level_file)
    log_level_console = entries_to_str(log_level_console)
    log_file_name = entries_to_str(log_file_name)
    log_file_dir = entries_to_str(log_file_dir)

    if log_file_dir is None:
        import raredecay.globals_
        log_file_dir = raredecay.globals_.out.get_logger_path()
        if not isinstance(log_file_dir, basestring):
            # set logging only to console; if 'file' was selected, no console,
            # set logging to console with level 'critical'
            if logging_mode == 'file':
                log_level_console = 'critical'
            logging_mode = 'console'

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    # may be changed due to performance issues, does not have to log everything
    logger.propagate = False
    file_mode = 'w' if overwrite_file else None
    formatter = logging.Formatter("%(asctime)s - " + module_name +
                                  ": %(levelname)s - %(message)s")
    if logging_mode == 'both' or logging_mode == 'file':
        if overwrite_file:
            time_stamp = 'logfile'
        else:
            time_stamp = strftime("%a-%d-%b-%Y-%H:%M:%S")
        log_file_dir += '' if log_file_dir.endswith('/') else '/'
        log_file_fullname = log_file_dir + log_file_name + module_name
        fh = logging.FileHandler('%s-%s-logfile.txt' % (log_file_fullname,
                                                        time_stamp), file_mode)
        fh.setLevel(getattr(logging, log_level_file.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if logging_mode == 'both' or logging_mode == 'console':
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level_console.upper()))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # add logger to the loggers collection
    meta_cfg.loggers[module_name] = logger
    logger.info('Logger created succesfully, loggers: ' + str(meta_cfg.loggers))
    return logger


def progress(n, n_tot):
    """A simple progress-bar.

    Parameters
    ----------
    n : int or float
        Shows how far it is already
    n_tot : int or float
        The maximum. The bar is the percentage of n / n_tot
    """
    i = float(n) / n_tot
    percent = int(i * 100)
    n_signs = 90
    equals = int(n_signs * i) * '='
    spaces = (n_signs - len(equals)) * ' '
    sys.stdout.write("\r[" + equals + spaces + " %d%%]" % percent)
    sys.stdout.flush()


def add_file_handler(logger, module_name, log_file_dir, log_level='info',
                     overwrite_file=False):
    """Add a filehandler to a logger to also direct the output to a file."""
    from time import strftime
    import logging

    logger = entries_to_str(logger)
    module_name = entries_to_str(module_name)
    log_file_dir = entries_to_str(log_file_dir)
    log_level = entries_to_str(log_level)

    file_mode = 'w' if overwrite_file else None
    formatter = logging.Formatter("%(asctime)s - " + module_name +
                                  ": %(levelname)s - %(message)s")

    if overwrite_file:
        time_stamp = 'logfile'
    else:
        time_stamp = strftime("%a-%d-%b-%Y-%H:%M:%S")
    log_file_dir += '' if log_file_dir.endswith('/') else '/'
    log_file_fullname = log_file_dir + module_name
    fh = logging.FileHandler('%s-%s-logfile.txt' % (log_file_fullname,
                                                    time_stamp), file_mode)
    fh.setLevel(getattr(logging, log_level.upper()))
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def check_var(variable, allowed_range, default=None, logger=None):
    """Check if a given variable (string, number etc.) is "allowed."""

    # Dictionary
    if variable not in allowed_range:
        logger.warning(str(variable) + " is not a valid choice of " +
                       str(list(allowed_range.keys())) +
                       ". Instead, the default value was used: " +
                       default)
        variable = default
    return variable


def fill_list_var(to_check, length=0, var=1):
    """Return a list filled with the specified variable and the desired length."""

    difference = length - len(to_check)
    if difference > 0:
        if isinstance(to_check, list):
            to_check.extend([var] * difference)
    return to_check


def make_list_fill_var(to_check, length=0, var=None):
    """Returns a list with the objects or a list filled with var."""
    if not isinstance(to_check, list):
        to_check = [to_check]
    difference = length - len(to_check)
    if difference > 0:
        to_check += [var] * difference
    return to_check


def is_in_primitive(test_object, allowed_primitives=None):
    """Actually "equivalent" to: 'test_object is allowed_primitives' resp.
    'test_object in allowed_primitives' but also works for arrays etc.

    There is a problem (designproblem?!) with certain container types when
    comparing them to a certain value, e.g. None and you create an expression
    with 'container is None' and the container (as it is a container) is not
    None, then the expression is not defined. Because the code tries to make
    elementwise comparison.
    This method allows to fix this error by doing a REAL comparison. E.g.
    if the test_object is ANY container and the allowed_primitives is None,
    it will always be False.

    Parameters
    ----------
    test_object : any python-object
        The object you want to test whether it is one of the
        allowed_primitives.
    allowed_primitives : object or list(obj, obj, obj,...)
        Has to be testable as 'is obj'. If test_object is any of the
        allowed_primitives, True will be returned.

    Return
    ------
    out : boolean
        Returns True if test_object is any of the allowed_primitives,
        otherwise False.
    """
    test_object = entries_to_str(test_object)
    flag = False
    if isinstance(test_object, (list, np.ndarray, pd.Series, pd.DataFrame)):
        flag = False
    elif (isinstance(allowed_primitives, collections.Iterable) and
              (not isinstance(allowed_primitives, basestring))):
        if test_object in allowed_primitives:
            flag = True
    elif test_object is allowed_primitives:
        flag = True
    return flag


def entries_to_str(data):
    """Convert each basestring entry of a basestring/dict/list into a str.

    Parameters
    ----------
    data : dict, list

    Returns
    -------
    dict, list, str
        Return the dict with the new entries.
    """
    if isinstance(data, basestring):
        output = str(data)
    elif isinstance(data, dict):
        output = {}
        for key, val in data.items():
            key = entries_to_str(key)
            val = entries_to_str(val)
            output[key] = val

    elif isinstance(data, list):
        output = [entries_to_str(d) for d in data]
    else:
        output = data

    return output


def play_sound(duration=0.3, frequency=440):
    """ Play a single frequency (Hertz) for a given time (seconds)."""
    freq = frequency
    import os
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
