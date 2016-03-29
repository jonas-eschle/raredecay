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


def make_logger(module_name, logging_mode='both', log_level_file='debug',
                log_level_console='debug', overwrite_file=True,
                log_file_name='AAlast_run',log_file_dir='.'):
    """Return a logger with a console-/filehandler or both.

    A useful tool to log the run of the program and debug or control it. With
    logger.debug("message") a loging message is produced consisting of:
    timestamp(from year to ms) - module_name - logger level - message
    This can be either written to files, the console or both.

    Parameters
    ----------
    module_name : string
        Name of the logger, shown in output. Best choose __name__
    logging_mode : {"both", "file", "console"}
        Which logger handler is used; where the log is printed to.
    log_level_file : {"debug","info","warning","error","critical"}
        Which level of messages are logged. A lower level (left) always also
        includes the higher (right) levels, but not the other way around.
        This level specifies the level for the file log (if enabled).
    log_level_console : {"debug","info","warning","error","critical"}
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

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    # may be changed due to performance issues, does not have to log everything
    logger.propagate = False
    file_mode = 'w' if overwrite_file else None
    formatter = logging.Formatter("%(asctime)s - " + module_name +
                                  ": %(levelname)s - %(message)s")
    print 2
    if logging_mode == 'both' or logging_mode == 'file':
        if not overwrite_file:
            timeStamp = strftime("%a-%d-%b-%Y-%H:%M:%S")
        else:
            timeStamp = 'temp'
        if log_file_dir[-1] not in ('/'):
            log_file_dir += '/'
        log_file_fullname = log_file_dir + log_file_name
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
