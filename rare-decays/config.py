# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:26:13 2016

@author: mayou
"""


# reweighting
path_mc_reweight = "../data/DarkBoson/Bu2K1ee-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root"
path_real_reweight = "../data/DarkBoson/Bu2K1Jpsi-mm-DecProdCut-MC-2012-MagAll-Stripping20r0p3-Sim08g-withMCtruth.root"
branch_names = ["B_PT", "nTracks"]
tree_mc_reweight = None # "DecayTree"
tree_real_reweight = None # "DecayTree"












pathes_to_add = []

# configure LOGGER
# -----------------------------------------------------------
loggerMode = 'both'   # define where the logger is written to
# take 'both', 'file', 'console' or 'no'
loggerLevelFile = 'debug'
# specifies the level to be logged to the file
loggerLevelConsole = 'debug'
# specifies the level to be logged to the console
loggerOverwrite = True
# specifies whether it should overwrite the log file each time
# or instead make a new one each run
loggerFileName = 'AAlastRun'
# the beginning ofthe name of the logfile, like 'project1'


def _selftest_system():
    """Test the configuration regarding the system-relevant parameters"""

    # test pathes_to_add
    if not all(type(i) == str for i in pathes_to_add):
        raise TypeError(str(filter(lambda i: type(i) != str, pathes_to_add)) +
                        " not of type string")
    # test loggerMode
    if loggerMode not in ("both", "file", "console"):
        raise ValueError(str(loggerMode) + ": invalid choice for loggerMode")

    # test loggerLevel


def test_all():
    _selftest_system()

if __name__ == "__main__":
    test_all()
    print "config file succesfully tested!"
