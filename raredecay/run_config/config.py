# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:44:43 2016

The configuration file for external operations.

@author: Jonas Eschle "Mayou36"
"""


RUN_NAME = 'Classifier optimization'
run_message = str("This could be your advertisement" +
                  " ")

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
    figsize=(2,10)
)


# ==============================================================================
# LOGGER CONFIGURATION BEGIN
# ==============================================================================
logger_cfg = dict(
    logging_mode='both',   # define where the logger is written to
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
