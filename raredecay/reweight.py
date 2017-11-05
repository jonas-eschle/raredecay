"""

@author: Jonas Eschle "Mayou36"

"""
# Python 2 backwards compatibility overhead START
from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct,  # noqa
                      open, pow, range, round, str, super, zip,
                      )  # noqa
import sys  # noqa
import warnings  # noqa
import raredecay.meta_config  # noqa

try:  # noqa
    from future.builtins.disabled import (apply, cmp, coerce, execfile, file, long, raw_input,  # noqa
                                          reduce, reload, unicode, xrange, StandardError,
                                          )  # noqa
    from future.standard_library import install_aliases  # noqa

    install_aliases()  # noqa
except ImportError as err:  # noqa
    if sys.version_info[0] < 3:  # noqa
        if raredecay.meta_config.SUPPRESS_FUTURE_IMPORT_ERROR:  # noqa
            raredecay.meta_config.warning_occured()  # noqa
            warnings.warn("Module future is not imported, error is suppressed. This means "  # noqa
                          "Python 3 code is run under 2.7, which can cause unpredictable"  # noqa
                          "errors. Best install the future package.", RuntimeWarning)  # noqa
        else:  # noqa
            raise err  # noqa
    else:  # noqa
        basestring = str  # noqa

# Python 2 backwards compatibility overhead END

try:
    from raredecay.analysis.reweight import reweight, reweight_kfold
except Exception as err:
    print("could not import reweighting algorithms  (missing deps?)", str(err))

__modules = ['reweight', 'reweight_kfold']
try:
    from raredecay.analysis.physical_analysis import reweightCV
except ImportError:
    __modules.append('reweightCV')

__all__ = __modules
