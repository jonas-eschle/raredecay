"""

@author: Jonas Eschle "Mayou36"

"""


# Python 2 backwards compatibility overhead START
import sys  # noqa
import warnings  # noqa
from . import meta_config  # noqa
from . import reweight

try:  # noqa
    from future.builtins.disabled import (
        apply,
        cmp,
        coerce,
        execfile,
        file,
        long,
        raw_input,  # noqa
        reduce,
        reload,
        unicode,
        xrange,
        StandardError,
    )  # noqa
    from future.standard_library import install_aliases  # noqa

    install_aliases()  # noqa
except ImportError as err:  # noqa
    if sys.version_info[0] < 3:  # noqa
        if meta_config.SUPPRESS_FUTURE_IMPORT_ERROR:  # noqa
            meta_config.warning_occured()  # noqa
            warnings.warn(
                "Module future is not imported, error is suppressed. This means "  # noqa
                "Python 3 code is run under 2.7, which can cause unpredictable"  # noqa
                "errors. Best install the future package.",
                RuntimeWarning,
            )  # noqa
        else:  # noqa
            raise err  # noqa
    else:  # noqa
        basestring = str  # noqa

# Python 2 backwards compatibility overhead END

from . import reweight, config, data, ml, score, meta_config
from . import settings, stat

if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    warnings.warn(
        f"UNSUPPORTED PYTHON VERSION: You are using {sys.version_info}, "
        + "which is an unsupported Python version < 3.6. "
        + "This is not tested nor guaranteed"
        + " to work and provided on an as-is based."
    )

__all__ = [
    "reweight",
    "config",
    "data",
    "ml",
    "score",
    "settings",
    "stat",
    "meta_config",
    "globals_",
]

__author__ = "Jonas Eschle 'Mayou36'"
__email__ = "Jonas.Eschle@cern.ch"
__version__ = "2.2.0"
