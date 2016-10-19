# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:45:51 2016

Contain methods to change settings in the whole package

@author: Jonas Eschle "Mayou36"
"""

from __future__ import division, absolute_import


import copy

from raredecay.run_config import config
from raredecay import meta_config


def initialize(output_path=None, run_name="Test run", overwrite_existing=False,
               run_message="This is a test-run to test the package", verbosity=3,
               plot_verbosity=3, prompt_for_input=False,
               logger_console_level='warning', logger_file_level='debug',
               n_cpu=1, gpu_in_use=False):
    """Place before Imports! Initialize/change several parameters for the package"""
    set_verbosity(verbosity=verbosity, plot_verbosity=plot_verbosity)
    _init_user_input(prompt_for_input=prompt_for_input)
    parallel_profile(n_cpu=n_cpu, gpu_in_use=gpu_in_use)
    logger_file_level = None if output_path is None else logger_file_level
    _init_configure_logger(console_level=logger_console_level,
                           file_level=logger_file_level)
    if output_path is not None:
        _init_output_to_file(file_path=output_path, run_name=run_name,
                             overwrite_existing=overwrite_existing,
                             run_message=run_message)
    else:
        _init_output_to_file(file_path=None, run_name=run_name,
                             prompt_for_input=prompt_for_input)


def finalize(show_plots=True, play_sound_at_end=False):
    """Finalize the run, (save figures and output) and return output

    Parameters
    ----------
    show_plots : boolean
        If True, show the plots (*plt.show()*), if user prompt is activated,
        you first have to press enter to show them.
    play_sound_at_end : boolean
        If true, a beep will be played at the end of the run
    """
    out = get_output_handler()
    output = out.finalize(show_plots=show_plots, play_sound_at_end=play_sound_at_end)
    return output


def set_verbosity(verbosity=3, plot_verbosity=3):
    """Change the verbosity of the package"""
    if verbosity is not None:
        meta_config.set_verbosity(verbosity)
    if plot_verbosity is not None:
        meta_config.set_plot_verbosity(plot_verbosity)


def get_output_handler():
    """Return an output handler, instance of :py:class:`~raredecay.tools.output.OutputHandler()`

    This can be used to add output (text as well as figures) and save them
    easely. For more information see the docs of the OutputHandler
    """
    from raredecay.globals_ import out
    return out


def parallel_profile(n_cpu=-1, gpu_in_use=False):
    """Set the parallel profile, the n_cpu to use

    Most of the functions do not have a number of threads option. You can
    change that here by either:

    - using a positive integer, which corresponds to the number of threads to
      use
    - using a negative integer. Then, -1 means to use all available (virtual)
      cores (resp. the number of threads). But sometimes you want to still work
      on your machine and have not all cpus used up. Therefore you can just use
      -2 (which will use all except of one; "all up to the second last"), -3
      (which will use all except two; "all up to the third last") etc.

    Parameters
    ----------
    n_cpu : int (not 0)
        The number of cpus to use, explanation see above.
    gpu_in_use : boolean
        If a gpu is in use (say for nn), you can set this boolean to true in
        order to avoid parallel computation of the nn or to activate any
        other implemented gpu assistance.


    """
    meta_config.set_parallel_profile(n_cpu=n_cpu, gpu_in_use=gpu_in_use)


def figure_save_config(file_formats=None, to_pickle=True, dpi=150):
    """Change the save-options of figures.

    If you initialized an output-path, the figures that are plotted during
    the run will be f. On one hand, they are saved as pictures in the
    given formats, on the other hand the figures (matplotlib) will be saved
    as a pickle-object (also in the output-folder)
    If the run was not initialized with an output-path, this function will
    have no effect on the behaviour of your script.

    Parameters
    ----------
    file_formats : str or list[str, str, str,...]
        The possible formats to save the figures to. Currently implemented are:
        ['png', 'jpg', 'pdf', 'svg']

        The default value (None) is ['png', 'svg']
    to_pickle : boolean
        If True, the matplotlib-figures will be saved as a pickle-file allowing
        for later plotting. They are saved in the output-folder.
    dpi : int
        The resolution of the images.
    """
    # hack for using mutable defaults
    file_formats = copy.deepcopy(file_formats)
    config.save_fig_cfg['file_formats'] = file_formats
    config.save_fig_cfg['to_pickle'] = to_pickle
    config.save_fig_cfg['dpi'] = dpi


def set_random_seed(seed=None):
    """Set the seed to the random generator to reproduce results

    Parameters
    ----------
    seed : int
        The seed for the random generator. If None, it won't change anything
    """
    if seed is not None:
        meta_config.set_seed(seed)


def _init_output_to_file(file_path, run_name="Test run", overwrite_existing=False,
                         run_message="This is a test-run to test the package",
                         prompt_for_input=False):
    """Saves output to file"""
    assert isinstance(run_name, (str, int)), "run_name has to be a string or int"
    config.RUN_NAME = str(run_name)
    config.OUTPUT_CFG['run_name'] = str(run_name)

    if file_path is not None:
        assert isinstance(file_path, str), "file_path has to be a string"

        file_path = str(file_path) if isinstance(file_path, int) else file_path
        file_path += "" if file_path.endswith("/") else "/"

        config.run_message = str(run_message)
        config.OUTPUT_CFG['output_path'] = file_path
        config.OUTPUT_CFG['del_existing_folders'] = overwrite_existing

        out = get_output_handler()
        out.initialize_save(logger_cfg=config.logger_cfg, **config.OUTPUT_CFG)
        out.make_me_a_logger()
    else:
        out = get_output_handler()
        out.initialize(run_name=run_name, prompt_for_comment=prompt_for_input)


def _init_configure_logger(console_level='critical', file_level='debug'):
    """Call before imports! Set the logger-level

    The package contains several loggers which will print/save to file some
    information. What kind of information is handled can be changed by this
    function. The higher (severer) the level, the less will be displayed.

    Parameters
    ----------
    console_level : None or str {'debug', 'info', 'warning', 'error', 'critical'}
        Define the logging-level of the console handler. None means no output
        at all.
    file_level : None or str {'debug', 'info', 'warning', 'error', 'critical'}
        Define the logging-level of the file handler. None means no output
        at all.
    """
    if console_level is None and file_level is None:
        logging_mode = 'file'
        console_level = 'critical'
    elif console_level is None:
        logging_mode = 'file'
    elif file_level is None:
        logging_mode = 'console'
    else:
        logging_mode = 'both'

    for level in (console_level, file_level):
        assert level in (None, 'debug', 'info', 'warning', 'error', 'critical'), \
            "invalid logger level"

    config.logger_cfg['logging_mode'] = logging_mode
    config.logger_cfg['log_level_file'] = file_level
    config.logger_cfg['log_level_console'] = console_level
    config.logger_cfg['overwrite_file'] = True
    config.logger_cfg['log_file_name'] = 'logfile_'


def _init_user_input(prompt_for_input=True):
    """If called, you will be asked for input to name the specific run"""
    meta_config.NO_PROMPT_ASSUME_YES = not prompt_for_input
    meta_config.PROMPT_FOR_COMMENT = prompt_for_input
