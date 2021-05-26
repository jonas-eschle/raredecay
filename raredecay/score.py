"""

@author: Jonas Eschle "Mayou36"

"""
# Python 2 backwards compatibility overhead START
import sys  # noqa
import warnings  # noqa
from . import meta_config  # noqa

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

try:
    from raredecay.tools.ml_scores import mayou_score, train_similar, train_similar_new
    from raredecay.analysis.ml_analysis import mcreweighted_as_real

    __all__ = [
        "train_similar_new",
        "train_similar",
        "mayou_score",
        "mcreweighted_as_real",
    ]

except Exception as err:
    print("could not import machine learning based scores (missing deps?)", str(err))
