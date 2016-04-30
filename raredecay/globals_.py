# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:34:21 2016

@author: mayou

This module contains all (package-)global variables and methods.

Variables:
---------
randint: int
    Many methods need random integers for their pseudo-random generator.
    To keep them all the same (or intentionally not), use the randint.
"""
from __future__ import division, absolute_import

import warnings
import sys
import subprocess
import os

import random

from raredecay.tools import dev_tool
from raredecay import meta_config


randint = random.randint(123, 1512412)  # 357422 or 566575


_output_path = None
_output_folders = None
_del_existing_dir = False
_path_to_be_overriden = None
output = ""
logger_path = None

def get_logger_path():
    """Return the path for the log folder"""

    global _output_path
    print _output_path
    global _output_folders
    print _output_folders
    path_out = _output_path + _output_folders.get('log')
    path_out += '' if path_out.endswith('/') else '/'
    return path_out


def add_output(data_out, title=None, obj_separator=None, data_separator=None,
               do_print=True, force_newline=True):
    """Method to collect the output and format

    Parameter
    ---------
    data_out : obj or list(obj, obj, obj, ...)
        The data to be added to the output. Has to be convertible to string!
    title : str
        The title of the data_out, like "roc auc of different classifiers".
        If None, no title will be set.
    obj_separator : str
        The separator between the objects in data_out.
        Default is a new line: "\n".
    data_separator : str
        | Separates the data_outs from each other. Inserted at the end and
        creates a separation from the next call of add_output.
        | Default is a blank line as separation: "\n\n".
    do_print : boolean
        If True, the data will not only be added to the output but
        also printed when add_output is called.
    force_newline : boolean
        If true, the data_out will be written on a new line and not just
        concatenated to the data written before
    """
    global output
    # initialize defaults
    obj_separator = "\n" if obj_separator is None else obj_separator
    data_separator = "\n\n" if data_separator is None else data_separator
    assert isinstance(obj_separator, str), (str(obj_separator) + " is of type " + str(type(obj_separator)) + " instead of string")
    assert isinstance(data_separator, str), (str(data_separator) + " is of type " + str(type(data_separator)) + " instead of string")

    data_out = dev_tool.make_list_fill_var(data_out)
    title_underline = "-"
    temp_out = ""

    # enforce new line
    if (len(output) > 0) and (not output.endswith("\n")):
        temp_out = "\n"

    # set title
    if title is not None:
        assert isinstance(title, str), ("Title is not a string but " + str(type(title)))
        temp_out += title + "\n"
        temp_out += title_underline * len(title) + "\n"

# TODO: add nice format for dictionaries
    for word in data_out:
        temp_out += str(word)
        temp_out += obj_separator if word is not data_out[-1] else data_separator
    if do_print:
        sys.stdout.write(temp_out)
    output += temp_out


def initialize(output_path, run_name=None, output_folders=None,
               del_existing_dir=False, logger_cfg=None, **config_kw):
    """Initializes the run. Creates the neccesary folders.

    Parameter
    ---------
    Best Practice: enter a whole config file

    output_path : str
        Absolute path to the folder where the run output folder will be
        created (named after the run) which will contain all the output
        folders (logfile, figures, output etc)
    output_folders : dict
        Contain the name of the folders for the different outputs. For the
        available keys
        see :py:const:`~raredecay.meta_config.__DEFAULT_OUTPUT_FOLDERS`.
    del_existing_dir : boolean
        If True, an already existing folder with the same name will be deleted.
        If False and the folder exists already, an exception will be raised.
    """
    global _path_to_be_overriden
    global _output_folders
    global _output_path
    global _del_existing_dir

    # create logger
    logger_cfg = {} if logger_cfg is None else logger_cfg
    logger_cfg = dict(meta_config.__DEFAULT_LOGGER_CFG, **logger_cfg)
    #dev_tool.make_logger(__name__, **logger_cfg)

    assert isinstance(output_path, str), "output_path not a string"
    output_folders = {} if output_folders is None else output_folders
    _output_folders = dict(meta_config.__DEFAULT_OUTPUT_FOLDERS, **output_folders)

    # find a non-existing folder
    run_name = str(run_name)
    output_path += run_name if output_path.endswith('/') else '/' + run_name
    output_path = os.path.expanduser(output_path)  # replaces ~ with /home/myUser
    _output_path = output_path

    temp_i = 1
    while os.path.isdir(_output_path):
        if del_existing_dir:
            _path_to_be_overriden = output_path
            _path_to_be_overriden += '' if _path_to_be_overriden.endswith('/') else '/'
        _output_path = output_path + "_" + str(temp_i)
        temp_i += 1
        assert temp_i < meta_config.MAX_AUTO_FOLDERS, "possible endless loop when trying to create a non-existing folder"
    _output_path += '' if output_path.endswith('/') else '/'

    #create subfolders
    for value in _output_folders.itervalues():
        assert isinstance(value, str), "path is not a string: " + str(value)
        subprocess.call(['mkdir', '-p', _output_path + value])

    # create the default log file in case it is needed
    subprocess.call(['mkdir', '-p', _output_path +
                    meta_config.__DEFAULT_LOGGER_CFG.get('log_file_dir')])


def finalize():
    """Finalize the run: save output and information to file.
    """
    # add the pseudo random-generator integer
    global _output_path
    global _path_to_be_overriden
    global randint
    global output

    add_output(["randint: ", randint], title="Different parameters",
               obj_separator=" - ")
    global commit_nr
    add_output(["Git commit number"])
#==============================================================================
#   write output to file
#==============================================================================

    temp_out_file = _output_path + _output_folders.get('results') + '/output.txt'
    with open(temp_out_file, 'w') as f:
        f.write(output)
    del temp_out_file  # block abuse

#==============================================================================
#    if a folder to overwrite exists, delete it and move the temp folder
#==============================================================================
    if _path_to_be_overriden is not None:
        if not meta_config.NO_PROMPT_ASSUME_YES:
            stop_del = raw_input("ATTENTION! The folder " + _path_to_be_overriden +
                        " will be deleted and replaced with the output of the current run." +
                        "\nTo DELETE that folder and overwrite, press ENTER.\n\n" +
                        "If you want to keep the folder and save the current run under " +
                        _output_path + ", please enter any input and press enter.\n\nYour input:")
        if stop_del == '':
            subprocess.call(['rm', '-r', _path_to_be_overriden])
            subprocess.call(['mv', _output_path, _path_to_be_overriden])
            path = _path_to_be_overriden
        else:
            path = _output_path
    else:
        path = _output_path
    print "All output saved under: " + path
    subprocess.call(['touch', path + '.finished'])  # .finished shows if the run finished



if __name__=='__main__':
    print "Selftest start"
    add_output("hello world just as test", title="A simple Hello World")
    add_output(["my age = 1", "we have unknowns", 42], title="some ages", data_separator="\n\n\n\n------")
    add_output(["i am a placeholder", "how am i separated?"], obj_separator=" - ")
    print "And this it the output variable:\n" + output
    print "Selftest completed"