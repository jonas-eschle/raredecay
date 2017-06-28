# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:25:26 2016

@author: Jonas Eschle "Mayou36"

Contains several useful tools for all kind of programs
"""
from __future__ import division, absolute_import

import pandas as pd
import numpy as np
import collections

from raredecay import meta_config


def syspath_append(verboise=False):
    """Adds the relevant path to the sys.path variable.
    options:
    v for verboise, print sys.paht before and after
    """
    import sys
    import config

    if verboise == 'v':
        verboise = True
    if verboise:
        print sys.path
    # n_to_remove = 0 #number of elements to remove from sys.path from behind
    # sys.path = sys.path[:len(sys.path)-n_to_remove]
    # used to remove unnecessary bindings
    for path in config.pathes_to_add:
        # get the sys.path and add pathes if they are not already contained
        if path not in sys.path:
            try:
                sys.path.append(path)
            except:
                print "error when adding path \"" + path + "\" to sys.path"
    if verboise:
        print sys.path


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

    if log_file_dir is None:
        import raredecay.globals_
        log_file_dir = raredecay.globals_.out.get_logger_path()
        if not isinstance(log_file_dir, str):
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
            timeStamp = 'logfile'
        else:
            timeStamp = strftime("%a-%d-%b-%Y-%H:%M:%S")
        log_file_dir += '' if log_file_dir.endswith('/') else '/'
        log_file_fullname = log_file_dir + log_file_name + module_name
        fh = logging.FileHandler('%s-%s-logfile.txt' % (log_file_fullname,
                                                        timeStamp), file_mode)
        fh.setLevel(getattr(logging, log_level_file.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if logging_mode == 'both' or logging_mode == 'console':
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level_console.upper()))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # add logger to the loggers collection
    meta_config.loggers[module_name] = logger
    logger.info('Logger created succesfully, loggers: ' + str(meta_config.loggers))
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
    import sys

    i = float(n)/n_tot
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

    file_mode = 'w' if overwrite_file else None
    formatter = logging.Formatter("%(asctime)s - " + module_name +
                                  ": %(levelname)s - %(message)s")

    if overwrite_file:
        timeStamp = 'logfile'
    else:
        timeStamp = strftime("%a-%d-%b-%Y-%H:%M:%S")
    log_file_dir += '' if log_file_dir.endswith('/') else '/'
    log_file_fullname = log_file_dir + module_name
    fh = logging.FileHandler('%s-%s-logfile.txt' % (log_file_fullname,
                                                    timeStamp), file_mode)
    fh.setLevel(getattr(logging, log_level.upper()))
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def check_var(variable, allowed_range, default=None, logger=None):
    """Check if a given variable (string, number etc.) is "allowed."""

    # Dictionary
    if variable not in allowed_range:
        logger.warning(str(variable) + " is not a valid choice of " +
                       str(allowed_range.keys()) +
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
        to_check += [var]*difference
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


def play_sound(duration=0.3, frequency=440):
    """ Play a single frequency (Hertz) for a given time (seconds)."""
    freq = frequency
    import os
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
