# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:06:06 2016

@author: Jonas Eschle "Mayou36"
"""
import os
import sys
import subprocess
import warnings
import timeit
import time
import cStringIO as StringIO
import copy

import matplotlib.pyplot as plt
import cPickle as pickle
import seaborn as sns

from raredecay import meta_config
from raredecay.tools import dev_tool  # , data_tools


class OutputHandler(object):

    """Class for output handling."""

    __SAVE_STDOUT = sys.stdout
    _IMPLEMENTED_FORMATS = set(['png', 'jpg', 'pdf', 'svg'])
    _MOST_REPLACE_CHAR = [' ', '-', '<', '>', '&', '!', '?', '=', '*', '%', '.']
    _REPLACE_CHAR = _MOST_REPLACE_CHAR + ['/']

    def __init__(self):
        """Initialize an output handler"""

        self.output = ""
        self.end_output = ""
        self._loud_end_output = ""
        self._IO_string = ""
        self.logger = None
        self._logger_cfg = None
        self._is_initialized = False
        self._save_output = False
        self._run_name = ""

        self._output_path = None
        self._output_folders = None
        self._path_to_be_overriden = None
        self._figures = {}
        self._formats_used = set([])
        self._pickle_folder = False

        # start timer and log current time
        self._start_timer = timeit.default_timer()
        self._start_time = time.strftime("%c")

        # set plotting style
        sns.set_context("poster")
        plt.rc('figure', figsize=(20, 20))

        setattr(self, 'print', self._print)

    def _check_initialization(self, return_error=False):
        if not self._is_initialized and not return_error:
            self.initialize()
        elif not self._is_initialized and return_error:
            raise Exception("OutputHandler not initialized! You have to initialize it first")

    def initialize_save(self, output_path, run_name="", run_message="", output_folders=None,
                        del_existing_folders=False, logger_cfg=None):
        """Initialize the run. Create the neccesary folders.

        Parameters
        ----------
        Best Practice: enter a whole config file

        output_path : str
            Absolute path to the folder where the run output folder will be
            created (named after the run) which will contain all the output
            folders (logfile, figures, output etc)
        run_name : str
            The name of the run and also of the output folder.
        run_message : str
            A message that is displayed below the titel: a further comment
            on what you do in the script
        output_folders : dict
            Contain the name of the folders for the different outputs. For the
            available keys
            see :py:const:`~raredecay.meta_config.__DEFAULT_OUTPUT_FOLDERS`.
        del_existing_dir : boolean
            If True, an already existing folder with the same name will be deleted.
            If False and the folder exists already, an exception will be raised.
        logger_cfg : dict
            The configuration for the logger, which will be created later. If
            not specified (or only a few arguments), the meta_config will be
            taken.
        """
        self._save_output = True
        # initialize defaults
        logger_cfg = {} if logger_cfg is None else logger_cfg
        self._logger_cfg = dict(meta_config.DEFAULT_LOGGER_CFG, **logger_cfg)

        assert isinstance(output_path, str), "output_path not a string"
        output_folders = {} if output_folders is None else output_folders
        self._output_folders = dict(meta_config.DEFAULT_OUTPUT_FOLDERS, **output_folders)

        # make sure no blank spaces are left in the folder names
        for key, value in self._output_folders.iteritems():
            assert isinstance(value, str), "path is not a string: " + str(value)
            self._output_folders[key] = value.replace(" ", "_")

        # ask if you want to add something to the run_name (and folder name)
        if meta_config.PROMPT_FOR_COMMENT:
            prompt_message = "Enter an (optional) extension to the run-name and press 'enter':\n"
            temp_add = str(raw_input(prompt_message))
            run_name += " " + temp_add if temp_add != "" else ""
            # del temp_add
            # TODO: implement promt with timeout
        self._run_name = run_name

        # "clean" and correct the path-name
        for char in self._REPLACE_CHAR:
            run_name = run_name.replace(char, "_")
        output_path += run_name if output_path.endswith('/') else '/' + run_name
        self._output_path = os.path.expanduser(output_path)  # replaces ~ with /home/myUser

        # find a non-existing folder
        temp_i = 1
        while os.path.isdir(self._output_path):
            if del_existing_folders:
                self._path_to_be_overriden = output_path
                if not self._path_to_be_overriden.endswith('/'):
                    self._path_to_be_overriden += '/'
            self._output_path = output_path + "_" + str(temp_i)
            temp_i += 1
            assert temp_i < meta_config.MAX_AUTO_FOLDERS, \
                "possible endless loop when trying to create a non-existing folder"
        self._output_path += '' if output_path.endswith('/') else '/'

        # create subfolders
        for value in self._output_folders.itervalues():
            subprocess.call(['mkdir', '-p', self._output_path + value])
        subprocess.call(['touch', self._output_path + 'run_NOT_finished'])  # show that ongoing run

        # set meta-config variables
        meta_config.set_parallel_profile(n_cpu=meta_config.n_cpu_max,
                                         gpu_in_use=meta_config.use_gpu)

        self._is_initialized = True
        self.add_output(run_message, title="Run: " + self._run_name, importance=0,
                        subtitle="Comments about the run")

    def initialize(self, run_name="", prompt_for_comment=False):
        """Initialization for external use, no folders created, config.py logger."""

        # initialize defaults
        from raredecay.globals_ import logger_cfg
        self._logger_cfg = logger_cfg
        self._is_initialized = True
        self.make_me_a_logger()
        # ask if you want to add something to the run_name (and folder name)
        if prompt_for_comment:
            prompt_message = "Enter an (optional) extension to the run-name and press 'enter':\n"
            temp_add = str(raw_input(prompt_message))
            run_name += " " + temp_add if temp_add != "" else ""
        self._run_name = str(run_name)

    def get_logger_path(self):
        """Return the path for the log folder."""
        if self._save_output:
            path_out = self._output_path + self._output_folders.get('log')
            path_out += '' if path_out.endswith('/') else '/'
        else:
            path_out = None
        return path_out

    def get_plots_path(self):
        """Return the path for the log folder."""
        if self._save_output:
            path_out = self._output_path + self._output_folders.get('plots')
            path_out += '' if path_out.endswith('/') else '/'
        else:
            path_out = None
        return path_out

    def make_me_a_logger(self):
        """Create a logger in OutputHandler instance. Dependency "hack".

        Call after :py:meth:`~raredecay.tools.output.OutputHandler.initialize_save()`
        has ben called.
        """
        # create logger
        self.logger = dev_tool.make_logger(__name__, **self._logger_cfg)

    def IO_to_string(self):
        """Rederict stdout (print etc.) to string."""
        self._IO_string = ""
        self._IO_string = StringIO.StringIO()
        sys.stdout = self._IO_string

    def IO_to_sys(self, importance=3, **add_output_kwarg):
        """Direct stdout back to the sys.stdout and return/save string to output.

        Parameters
        ----------
        importance : int {0, 1, 2, 3, 4, 5}
            | The importance of the output. The higher, the more likely it will
            | be added to the output. To not add it at all but only rederict
            | the output, choose 0.
            | Additional keyword-arguments for the
            | :py:meth:`~raredecay.tools.output.add_output()` method can be
            | passed.

        Return
        ------
        out : str
            Returns the collected string from the redirection.
        """
        sys.stdout = self.__SAVE_STDOUT
        self.add_output(self._IO_string.getvalue(), importance=importance, **add_output_kwarg)
        return self._IO_string.getvalue()

    def figure(self, *args, **kwargs):
        """FUTURE: Wrapper around save_fig()."""
        return self.save_fig(*args, **kwargs)

    def save_fig(self, figure, importance=3, file_format=None, to_pickle=True,
                 figure_kwargs=None, **save_cfg):
        """Advanced :py:meth:`matplotlib.pyplot.figure()`. Create and save a
        certain figure at the end of the run.

        To create and save a figure, you just enter an already created or a new
        figure as a Parameters and specify the fileformats it should be saved
        to. The figure can also be pickled so that it can be re-plotted
        anytime.


        .. note:: The figure will be saved at the end of the run
            (by calling :py:meth:`~raredecay.tools.output.OutputHandler.finalize`)
            so any change you make until the end will be applied to the plot.

        Parameters
        ----------
        figure : instance of :py:class:`matplotlib.figure.Figure` (e.g. returned \
                                                                   by :func:`matplotlib.figure`)
            The figure to be saved.
        importance : {0, 1, 2, 3, 4, 5}
            Specify the importance level, ranging from 1 to 5, of the plot.
            The higher the importance level, the more important.
            If the importance level is *higher*
            the more it will be plotted. If it is plotted depends on the
            plot verbosity set (5 - importance_level < plot_verbosity).
            Therefore, a 0 corresponds to "no plot" and a 5 means "always plot".
        file_format : str or list(str, str, str,...)
            The ending of the desired format, example: 'png' (default).
            If you don't want to save it, enter a blank list.
        to_pickle : boolean
            If True, the plot will be saved to a pickle file.
        **save_cfg : keyword args
            Will be used as arguments in :py:func:`~matplotlib.pyplot.savefig()`

        Return
        ------
        out : :py:class:`~matplotlib.pyplot.figure`
            Return the figure.
        """
        plot = 5 - round(importance) < meta_config.plot_verbosity  # to plot or not to plot
        figure_kwargs = {} if figure_kwargs is None else figure_kwargs

        if self._save_output:
            self._pickle_folder = self._pickle_folder or to_pickle
            if isinstance(figure, (int, str)):
                figure = plt.figure(figure, **figure_kwargs)  # TODO: changeable?

            file_format = meta_config.DEFAULT_SAVE_FIG['file_format'] if file_format is None else file_format
            if isinstance(file_format, str):
                file_format = [file_format]
            file_format = set(file_format)
            file_format.intersection_update(self._IMPLEMENTED_FORMATS)
            self._formats_used.update(file_format)

            # change layout of figures
#            figure.tight_layout()
#            figure.set_figheight(20)
#            figure.set_figwidth(20)

            # add figure to dict for later output to file
            figure_dict = {'figure': figure, 'file_format': file_format,
                           'to_pickle': to_pickle, 'plot': plot, 'save_cfg': save_cfg}
            self._figures[figure.get_label()] = figure_dict
        else:
            self._check_initialization()
            if plot and isinstance(figure, (int, str)):
                figure = plt.figure(figure, **figure_kwargs)

        return figure

    def _figure_to_file(self):
        """Write all figures from the _figures dictionary to file."""

        # check if there are figures to plot, else return
        if self._figures == {}:
            self.logger.info("_figure_to_file called but nothing to save/plot")
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
            for char in self._REPLACE_CHAR:
                fig_name = fig_name.replace(char, "_")
            for extension in fig_dict.get('file_format'):
                file_path = path + extension + '/'
                file_name = file_path + fig_name + "." + extension
                try:
                    figure_tmp = fig_dict['figure']
#                    figure_tmp.tight_layout()
                    figure_tmp.savefig(file_name, format=extension,
                                       **fig_dict.get('save_cfg'))
                except:
                    self.logger.error("Could not save figure" + str(figure_tmp))
                    meta_config.error_occured()

            if fig_dict.get('to_pickle'):
                file_name = (path + meta_config.PICKLE_DATATYPE + '/' +
                             fig_name + "." + meta_config.PICKLE_DATATYPE)
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

    @staticmethod
    def _make_title(title, title_format):
        """Create a title/subtitle/section in the reST-format and return it as
        a string.

        Parameters
        ----------
        title : str
            The title in words
        title_format : (str, str)
            | The surrounding lines. The titel will be:
            |
            | title_format[0] * len(title)
            | title
            | title_format[1] * len(title)
        """
        out_str = ""
        if title is not None:
            title = str(title)
            out_str += "\n" + title_format[0] * len(title)
            out_str += "\n" + title
            out_str += "\n" + title_format[1] * len(title) + "\n"
        return out_str

    def _print(self, data, to_end=False, importance=3, title=None,
               subtitle=None, section=None, obj_separator=" ",
               data_separator="\n\n", force_newline=True):

        return self.add_output(data_out=data, to_end=to_end, importance=importance, title=title,
                               subtitle=subtitle, section=section, obj_separator=obj_separator,
                               data_separator=data_separator, force_newline=force_newline)

    def add_output(self, data_out, to_end=False, importance=3, title=None,
                   subtitle=None, section=None, obj_separator=" ",
                   data_separator="\n\n", force_newline=True):
        """A method to collect the output and format it nicely.

        All the objects in data_out get converted to strings and concatenated
        with obj_separator in between. After the objects, a data_separator is
        added. In the end, the whole output gets printed to a file and saved.

        Available options:
            - You can add the data at the end of the output file instead of
              right in place.

            - You can add the data to the output "silently", without printing.

            - Add title, subtitle and section on top of the data.

        Parameters
        ----------
        data_out : obj or list(obj, obj, obj, ...)
            The data to be added to the output. Has to be convertible to str!
        to_end : boolean
            If True, the data will be added at the end of the file and not
            printed. For example all information which is not interesting in
            the run but maybe later, like configuration, version number etc.
        importance : int {0, 1, 2, 3, 4, 5}
            The importance of the output. The higher, the more likely it gets
            printed (otherwise only saved). A 0 means "don't print, only save".
            The decisive variable is the verbosity level. The lower the
            verbosity level, the less likely the output will be printed.
        title : str
            The title of the data_out, like "roc auc of different classifiers".
            If None, no title will be set.
        subtitle : str
            A subtitle which can be additional to a title or exclusive.
        section : str
            The section title. Can be additional to the others or exclusive.
        obj_separator : str
            The separator between the objects in data_out.
            Default is a new line.
        data_separator : str
            Separates the data_outs from each other. Inserted at the end and
            creates a separation from the next call of add_output.
            Default is a blank line as separation.
        force_newline : boolean
            If true, the data_out will be written on a new line and not just
            concatenated to the data written before
        """
        # initialize defaults
        assert isinstance(obj_separator, str), \
            (str(obj_separator) + " is of type " + str(type(obj_separator)) + " instead of string")
        assert isinstance(data_separator, str), \
            (str(data_separator) + " is of type " + str(type(data_separator)) + " instead of string")
        self._check_initialization()
        do_print = 5 - round(importance) < meta_config.verbosity

        data_out = dev_tool.make_list_fill_var(data_out)
        temp_out = ""

        # enforce new line
        if (len(self.output) > 0) and (not self.output.endswith("\n")):
            temp_out = "\n" if force_newline else ""

        # set title, subtitle and section with title_format, subtitle_format...
        title_f = ('=', '=')
        subtitle_f = ('-', '-')
        section_f = ('', '=')
        for name, form in ((title, title_f), (subtitle, subtitle_f), (section, section_f)):
            temp_out += self._make_title(name, form)

        # Concatenation of the objects
        for word in data_out:
            # Make nice format for dictionaries
            if isinstance(word, dict):
                for key, val in word.iteritems():
                    if isinstance(val, dict):
                        temp_out += self._make_title("  " + key, ('', '^'))
                        for key2, val2 in val.iteritems():
                            temp_out += "    " + str(key2) + " : " + str(val2) + "\n"
                    else:
                        sepr = "" if temp_out.endswith("\n") else "\n"
                        temp_out += sepr + "  " + str(key) + " : " + str(val)
            else:
                temp_out += str(word)
            temp_out += obj_separator if word is not data_out[-1] else data_separator

        # print and add to output collector
        if do_print:
            if to_end:
                self._loud_end_output += temp_out
            print temp_out  # ?? why was there sys.stdout.write?!?
        if to_end:
            self.end_output += temp_out
        self.output += temp_out

    def finalize(self, show_plots=True, play_sound_at_end=False):
        """Finalize the run. Save everything and plot.

        Parameters
        ----------
        show_plots : boolean
            If True, show the plots. Equivalent to writing plt.show().
        play_sound_at_end : boolean
            If True, tries to play a beep-sound at the end of a run to let you
            know it finished.
        """

        # ==============================================================================
        #  write all the necessary things to the output
        # ==============================================================================

        self.add_output("\n", title="END OF RUN " + self._run_name, importance=4)
        self.add_output(["Random generator seed", meta_config.rand_seed],
                        title="Different parameters", obj_separator=" : ", importance=2)

        # print the output which should be printed at the end of the run
        sys.stdout.write(self._loud_end_output)
        self.output += self.end_output

        # add current version (if available)
        if self._save_output and os.path.isdir(meta_config.GIT_DIR_PATH):
            try:
                git_version = subprocess.check_output(["git", "-C", meta_config.GIT_DIR_PATH,
                                                       "describe"])
                self.add_output(["Program version from Git", git_version],
                                section="Git information",
                                importance=0, obj_separator=" : ")
            except:
                meta_config.error_occured()
                self.logger.error("Could not get version number from git")

        # time information
        elapsed_time = timeit.default_timer() - self._start_timer
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        self.add_output(["Run startet at", self._start_time, "\nand lasted for",
                         elapsed_time], section="Time information", obj_separator=" ")

        # error information
        self.add_output(["Errors encountered during run", meta_config._error_count],
                        obj_separator=" : ")
        self.add_output(["Warnings encountered during run", meta_config._warning_count],
                        obj_separator=" : ")

        output = copy.deepcopy(self.output)

# ==============================================================================
#       save output to file
# ==============================================================================
        if self._save_output:

            # save figures to file
            self._figure_to_file()

            # Write output to file
            # ---------------------

            # remove leading blank lines
            for i in xrange(1, 100):
                if not self.output.startswith("\n" * i):  # "break" condition
                    self.output = self.output[i - 1:]
                    break

            temp_out_file = self._output_path + self._output_folders.get('results') + '/output.txt'
            try:
                with open(temp_out_file, 'w') as f:
                    f.write(self.output)
            except:
                self.logger.error("Could not save output to file")
                meta_config.error_occured()
                warnings.warn("Could not save output. Check the logs!", RuntimeWarning)

            # if a folder to overwrite exists, delete it and move the temp folder
            if self._path_to_be_overriden is not None:
                stop_del = ''
                if not meta_config.NO_PROMPT_ASSUME_YES:
                    stop_del = raw_input("ATTENTION! The folder " + self._path_to_be_overriden +
                                         " will be deleted and replaced with the output " +
                                         "of the current run.\nTo DELETE that folder and " +
                                         "overwrite, press ENTER.\n\nIf you want to keep the " +
                                         "folder and save the current run under " +
                                         self._output_path + ", please enter any input " +
                                         "and press enter.\n\nYour input:")
                if stop_del == '':
                    subprocess.call(['rm', '-r', self._path_to_be_overriden])
                    subprocess.call(['mv', self._output_path, self._path_to_be_overriden])
                    path = self._path_to_be_overriden
                else:
                    path = self._output_path
            else:
                path = self._output_path
            print "All output saved under: " + path
            subprocess.call(['rm', path + 'run_NOT_finished'])
            # .finished shows if the run finished
            subprocess.call(['touch', path + 'run_finished_succesfully'])

        self.output = self._loud_end_output = self.end_output = ""

        if play_sound_at_end:
            try:
                from raredecay.tools.dev_tool import play_sound
                play_sound()
            except:
                print "BEEEEEP, no sound could be played"

        if show_plots:
            if not meta_config.NO_PROMPT_ASSUME_YES:
                raw_input(["Run finished, press Enter to show the plots"])
            plt.show()

        return output
