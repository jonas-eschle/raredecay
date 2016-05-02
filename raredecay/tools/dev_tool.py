# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:25:26 2016

@author: mayou

Contains several useful tools for all kind of programs
"""
from __future__ import division, absolute_import

import numpy as np
import collections

def syspath_append(verboise=False):
    """Adds the relevant path to the sys.path variable.
    options:
    v for verboise, print sys.paht before and after
    """
    import sys
    import config

    if verboise == 'v': verboise = True
    if verboise: print sys.path
    # n_to_remove = 0 #number of elements to remove from sys.path from behind
    # sys.path = sys.path[:len(sys.path)-n_to_remove]
    # used to remove unnecessary bindings
    for path in config.pathes_to_add:
        """get the sys.path and add pahtes if they are not already contained"""
        if path not in sys.path:
            try:
                sys.path.append(path)
            except:
                print "error when adding path \""+path+"\" to sys.path"
    if verboise: print sys.path


def make_logger(module_name, logging_mode='both', log_level_file='debug',
                log_level_console='debug', overwrite_file=True,
                log_file_name='AAlast_run',log_file_dir=None):
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
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    # may be changed due to performance issues, does not have to log everything
    logger.propagate = False
    file_mode = 'w' if overwrite_file else None
    formatter = logging.Formatter("%(asctime)s - " + module_name +
                                  ": %(levelname)s - %(message)s")
    if logging_mode == 'both' or logging_mode == 'file':
        if overwrite_file:
            timeStamp = 'temp'
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

    logger.info('Logger created succesfully')
    return logger


def check_var(variable, allowed_range, default=None, logger=None):
    """Check if a given variable (string, number etc.) is "allowed"
    """

    # Dictionary
    if variable not in allowed_range:
        logger.warning(str(variable) + " is not a valid choice of " +
                            str(allowed_range.keys()) +
                            ". Instead, the default value was used: " +
                            default)
        variable = default
    return variable


def fill_list_var(to_check, length=0, var=1):
    """Returns a list filled with the specified variable and the desired length
    """
    difference = length - len(to_check)
    if difference > 0:
        if isinstance(to_check, list):
            to_check.extend([var] * difference)
    return to_check


def make_list_fill_var(to_check, length=0, var=None):
    """Returns a list with the objects or a list filled with None.
    """
    if not isinstance(to_check, list):
        to_check = [to_check]
    difference = length - len(to_check)
    if difference > 0:
        to_check += [var]*difference
    return to_check


def is_in_primitive(test_object, allowed_primitives):
    """Fixes the numpy/python "bug/stupidity" that ("==" can be replaced by
    "is"): "array([1,4,5]) == None" is not defined (it is clearly False)
    This way you can test safely for a primitive type. If the object is a list
    , array or similar, it returns 'False'.
    """
    flag = False
    if isinstance(test_object, (list, np.ndarray)):
        flag = False
    elif (isinstance(allowed_primitives, collections.Iterable) and
            (not isinstance(allowed_primitives, basestring))):
        if test_object in allowed_primitives:
            flag = True
    elif test_object is allowed_primitives:
        flag = True
    return flag



def play_sound(duration = 0.3, frequency = 440, change=False):
    """ Play a single frequency (Hertz) for a given time (seconds).
    """
    import os
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, frequency))
