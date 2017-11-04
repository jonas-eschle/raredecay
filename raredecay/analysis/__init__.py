"""
DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat


DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!

"""

# init of package analysis

# Python 2 backwards compatibility overhead START
from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct,
                      open, pow, range, round, str, super, zip,
                      )



# import raredecay.meta_config as meta_cfg


# if not meta_config.SUPPRESS_WRONG_SKLEARN_VERSION:
#    try:
#        temp_sklearn_info = imp.find_module('sklearn')
#        temp_sklearn = imp.load_module('sklearn', *temp_sklearn_info)
#        imp.find_module('cross_validation', temp_sklearn.__path__)
#    except ImportError:
#        try:
#            temp_sklearn_info = imp.find_module('sklearn')
#            temp_sklearn = imp.load_module('sklearn', *temp_sklearn_info)
#            imp.find_module('model_selection', temp_sklearn.__path__)
#        except ImportError:
#            print ("Could not import modules 'sklearn.cross_validation' or" +
#                   "'sklearn.model_selection'.\nIs scikit-learn installed?")
#            raise ImportError
#        else:
#            print str("Could not find module 'sklearn.cross_validation', but " +
#            "module 'sklearn.model_selection'.\nDo you have scikit-learn " +
#            ">= 0.18 installed? \nThis package is written for <=0.17. \n" +
#            "If you want to suppress this error, change the right flag in " +
#            "meta_config.py\n(strongly depreceated! Some parts will be " +
#            "unusable and raise errors)")
#            raise ImportError
