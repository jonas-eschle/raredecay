"""

@author: Jonas Eschle "Mayou36"

"""

# Python 2 backwards compatibility overhead START
import sys  # noqa
import warnings  # noqa

from . import (
    meta_config,
    reweight,
    reweight,
    config,
    data,
    ml,
    score,
    meta_config,
    settings,
    stat,
)  # noqa

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
