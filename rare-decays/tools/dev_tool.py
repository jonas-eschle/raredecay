# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:25:26 2016

@author: mayou
"""

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
    # ys.path = sys.path[:len(sys.path)-n_to_remove]
    # used to remove unnecessary bindings
    for path in config.pathes_to_add:
        """get the sys.path and add pahtes if they are not already contained"""
        if path not in sys.path:
            try:
                sys.path.append(path)
            except:
                print "error when adding path \""+path+"\" to sys.path"
    if verboise: print sys.path


def make_logger(moduleName, loggingMode='both', logLvlFile='debug',
                logLvlCons='debug', overwriteFile=True,
                logFileName='AAlast_run'):
    """Return a logger with a console-/filehandler or both.

    options:
    loggingMode
    """

    import logging
    from time import strftime

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # may be changed due to performance issues, does not have to log everything
    logger.propagate = False
    fileMode = 'w' if overwriteFile else None
    formatter = logging.Formatter("%(asctime)s - " + moduleName +
                                  ": %(levelname)s - %(message)s")
    if loggingMode == 'both' or loggingMode == 'file':
        if not overwriteFile:
            timeStamp = strftime("%a-%d-%b-%Y-%H:%M:%S")
        else:
            timeStamp = 'temp'
        fh = logging.FileHandler('%s-%s-logfile.txt' % (logFileName,
                                                        timeStamp), fileMode)
        fh.setLevel(getattr(logging, logLvlFile.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if loggingMode == 'both' or loggingMode == 'console':
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, logLvlCons.upper()))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info('Logger created succesfully')
    return logger
