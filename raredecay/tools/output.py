# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:06:06 2016

@author: mayou
"""
import os, sys
import subprocess

import matplotlib.pyplot as plt
import cPickle as pickle

from raredecay import meta_config
from raredecay.tools import dev_tool


class OutputHandler(object):
    """A class that can handle different kind of outputs and save them to file.

    """
    IMPLEMENTED_FORMATS = set(['png', 'jpg', 'pdf', 'svg'])

    def __init__(self):
            self._output_path = None
            self._output_folders = None
            self._path_to_be_overriden = None
            self.output = ""
            self.end_output = ""
            self.logger = None
            self._logger_cfg = None
            self._figures = {}
            self._formats_used = set([])
            self._pickle_folder = False

    def get_logger_path(self):
        """Return the path for the log folder"""
        path_out = self._output_path + self._output_folders.get('log')
        path_out += '' if path_out.endswith('/') else '/'
        return path_out

    def get_plots_path(self):
        """Return the path for the log folder"""
        path_out = self._output_path + self._output_folders.get('plots')
        path_out += '' if path_out.endswith('/') else '/'
        return path_out

    def make_me_a_logger(self):
        """Create a logger in OutputHandler instance. Dependency "hack". Call
        after :py:meth:`~raredecay.tools.output.OutputHandler.initialize()`
        has ben called.
        """
        # create logger
        self.logger = dev_tool.make_logger(__name__, **self._logger_cfg)

    def save_fig(self, figure, file_format=None, to_pickle=True, plot=True, **save_cfg):
        """Save a certain figure at the end of the run.

        To save a figure, you just hand it over as a parameter and specify the
        fileformats it should be saved to. The figure can also be pickled
        so that it can be re-plotted anytime.

        .. note:: The figure will be saved at the end of the run
            (by calling :py:meth:`~raredecay.tools.output.OutputHandler.finalize`)
            so any change you made until the end will be applied to the plot.

        Parameter
        ---------
        figure : instance of :class:`matplotlib.figure.Figure` (e.g. returned
        by :func:`matplotlib.figure`)
            The figure to be saved.
        file_format : str or list(str, str, str,...)
            The ending of the desired format, example: 'png' (default).
            If you don't want to save it, enter a blank list.
        to_pickle : boolean
            If True, the plot will be saved to a pickle file.
        plot : boolean
            If True, the figure will be showed when calling *show()*. If
            False, the figure will only be saved but not plotted.
        **save_cfg : keyword args
            Will be used as arguments in :py:func:`~matplotlib.pyplot.savefig()`
        """
        self._pickle_folder = self._pickle_folder or to_pickle

        file_format = ['png'] if file_format is None else file_format
        if isinstance(file_format, str):
            file_format = [file_format]
        file_format = set(file_format)
        file_format.intersection_update(self.IMPLEMENTED_FORMATS)
        self._formats_used.update(file_format)
        figure_dict = {'figure': figure, 'file_format': file_format,
                       'to_pickle': to_pickle, 'plot': plot, 'save_cfg': save_cfg}
        self._figures[figure.canvas.get_window_title()] = figure_dict

    def _figure_to_file(self):
        """Write all figure from the _figures dictionary to file"""

        # check if there are figures to plot, else return
        if self._figures == {}:
            self.logger.info("_figure to file called but nothing to plot")
            return None

        # create folders if they don't exist already
        path = self.get_plots_path()
        for format_ in self._formats_used:
            assert isinstance(format_, str), "Format is not a string: " + str(format_)
            subprocess.call(['mkdir', '-p', path + format_])
        if self._pickle_folder:
            subprocess.call(['mkdir', '-p', path + meta_config.PICKLE_DATATYPE])

        # save figures to file
        for fig_name, fig_dict in self._figures.iteritems():
            for extension in fig_dict.get('file_format'):
                file_path = path + extension + '/'
                file_name = file_path + fig_name + "." + extension
                file_name = file_name.replace(" ", "_")  # it was human-readable
                try:
                    fig_dict['figure'].savefig(file_name, format=extension,
                                               **fig_dict.get('save_cfg'))
                except:
                    self.logger.error("Could not save figure")
                    meta_config.error_occured()

            if fig_dict.get('to_pickle'):
                file_name = (path + meta_config.PICKLE_DATATYPE + '/' +
                             fig_name + "." + meta_config.PICKLE_DATATYPE)
                file_name = file_name.replace(" ", "_")
                try:
                    with open(str(file_name), 'wb') as f:
                        pickle.dump(fig_dict.get('figure'), f, meta_config.PICKLE_PROTOCOL)
                except:
                    self.logger.error("Could not open file" + str(file_name) +
                                      " OR pickle the figure to it")
                    meta_config.error_occured()

            # delete if it is not intended to be plotted
            if not fig_dict.get('plot'):
                plt.close(fig_dict.get('figure'))
        # clear the _figures dict
        self._figures = {}

    def add_output(self, data_out, to_end=False, title=None, subtitle=None,
                   section=None, obj_separator=None, data_separator=None,
                   do_print=True, force_newline=True):
        """Method to collect the output and format

        Parameter
        ---------
        data_out : obj or list(obj, obj, obj, ...)
            The data to be added to the output. Has to be convertible to string!
        to_end : boolean
            If True, the data will be added at the end of the file and not
            printed. For example all information which is not interesting in
            the run but maybe later, like configuration, version number etc.
        title : str
            The title of the data_out, like "roc auc of different classifiers".
            If None, no title will be set.
        subtitle : str
            A subtitle which can be additional to a title or exclusive.
        section : str
            The section title. Can be additional to the others or exclusive.
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
        # initialize defaults
        obj_separator = "\n" if obj_separator is None else obj_separator
        data_separator = "\n\n" if data_separator is None else data_separator
        assert isinstance(obj_separator, str), (str(obj_separator) + " is of type " + str(type(obj_separator)) + " instead of string")
        assert isinstance(data_separator, str), (str(data_separator) + " is of type " + str(type(data_separator)) + " instead of string")

        data_out = dev_tool.make_list_fill_var(data_out)
        temp_out = ""

        # enforce new line
        if (len(self.output) > 0) and (not self.output.endswith("\n")):
            temp_out = "\n"

        # set title, subtitle and section with title_format, subtitle_format...
        title_f = ('=', '=')
        subtitle_f = ('-', '-')
        section_f = ('', '=')
        for name, form in ((title, title_f), (subtitle, subtitle_f), (section, section_f)):
            if name is not None:
                name = str(name)
                temp_out += "\n" + form[0] * len(name)
                temp_out += "\n" + name
                temp_out += "\n" + form[1] * len(name) + "\n"

    # TODO: add nice format for dictionaries
        for word in data_out:
            temp_out += str(word)
            temp_out += obj_separator if word is not data_out[-1] else data_separator
        if do_print and (not to_end):
            sys.stdout.write(temp_out)
        if to_end:
            self.end_output += temp_out
        else:
            self.output += temp_out

    def initialize(self, output_path, run_name=None, output_folders=None,
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
        logger_cfg = {} if logger_cfg is None else logger_cfg
        self._logger_cfg = dict(meta_config.DEFAULT_LOGGER_CFG, **logger_cfg)

        assert isinstance(output_path, str), "output_path not a string"
        output_folders = {} if output_folders is None else output_folders
        self._output_folders = dict(meta_config.DEFAULT_OUTPUT_FOLDERS, **output_folders)

        # make sure no blank spaces are left in the folder names
        for key, value in self._output_folders.iteritems():
            assert isinstance(value, str), "path is not a string: " + str(value)
            self._output_folders[key] = value.replace(" ", "_")


        # find a non-existing folder
        run_name = str(run_name).replace(" ", "_")
        output_path += run_name if output_path.endswith('/') else '/' + run_name
        output_path = os.path.expanduser(output_path)  # replaces ~ with /home/myUser
        self._output_path = output_path.replace(" ", "_")

        temp_i = 1
        while os.path.isdir(self._output_path):
            if del_existing_dir:
                self._path_to_be_overriden = output_path
                self._path_to_be_overriden += '' if self._path_to_be_overriden.endswith('/') else '/'
            self._output_path = output_path + "_" + str(temp_i)
            temp_i += 1
            assert temp_i < meta_config.MAX_AUTO_FOLDERS, "possible endless loop when trying to create a non-existing folder"
        self._output_path += '' if output_path.endswith('/') else '/'

        # create subfolders
        for value in self._output_folders.itervalues():
            subprocess.call(['mkdir', '-p', self._output_path + value])

        # TODO: remove?:
        # create the default log file folder in case it is needed
        #subprocess.call(['mkdir', '-p', self._output_path +
         #               meta_config.DEFAULT_LOGGER_CFG.get('log_file_dir')])

    def finalize(self):
        """Finalize the run: save output and information to file.
        """
        from raredecay.globals_ import randint  # here to avoid circular dependencies

        # add the pseudo random-generator integer
        self.add_output(["randint", randint], title="Different parameters",
                        obj_separator=" : ", do_print=False)

        self.add_output("\n\n", do_print=False)
        self.add_output("\n\n", title="END OF RUN", do_print=False)
        self.output += self.end_output
        # TODO: get git informations
        git_version = subprocess.check_output(["git", "-C",
            "/home/mayou/Documents/uniphysik/Bachelor_thesis/python_workspace/HEP-decay-analysis/raredecay",
            "describe"])

        self.add_output(["Program version from Git", git_version], section="Git information",
                        do_print=False, obj_separator=" : ")
#==============================================================================
#   write output to file
#==============================================================================
        # remove leading blank lines
        for i in range(1,100):
            if not self.output.startswith("\n" * i):  # "break" condition
                self.output = self.output[i-1:]
                break

        temp_out_file = self._output_path + self._output_folders.get('results') + '/output.txt'
        try:
            with open(temp_out_file, 'w') as f:
                f.write(self.output)
        except:
            self.logger.error("Could not save output to file")
        #del temp_out_file  # block abuse

#==============================================================================
#  save figures to file
#==============================================================================

        self._figure_to_file()

#==============================================================================
#   copy the config file and save
#==============================================================================
       # TODO: copy config file. Necessary?


#==============================================================================
#    if a folder to overwrite exists, delete it and move the temp folder
#==============================================================================
        if self._path_to_be_overriden is not None:
            if not meta_config.NO_PROMPT_ASSUME_YES:
                stop_del = raw_input("ATTENTION! The folder " + self._path_to_be_overriden +
                            " will be deleted and replaced with the output of the current run." +
                            "\nTo DELETE that folder and overwrite, press ENTER.\n\n" +
                            "If you want to keep the folder and save the current run under " +
                            self._output_path + ", please enter any input and press enter.\n\nYour input:")
            if stop_del == '':
                subprocess.call(['rm', '-r', self._path_to_be_overriden])
                subprocess.call(['mv', self._output_path, self._path_to_be_overriden])
                path = self._path_to_be_overriden
            else:
                path = self._output_path
        else:
            path = self._output_path
        print "All output saved under: " + path
        subprocess.call(['touch', path + '.finished'])  # .finished shows if the run finished
