# -*- coding: utf-8 -*-
"""

@author: Jonas Eschle "Mayou36"

DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat

DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!

"""

# Python 2 backwards compatibility overhead START
from __future__ import absolute_import, division, print_function, unicode_literals

import sys  # noqa
import warnings  # noqa
from builtins import (dict, int, map, range, round, str)  # noqa

import raredecay.meta_config  # noqa

try:  # noqa
    from future.builtins.disabled import (apply, cmp, coerce, execfile, file, long, raw_input,  # noqa
                                          reduce, reload, unicode, xrange, StandardError,
                                          )  # noqa
    from future.standard_library import install_aliases  # noqa

    install_aliases()  # noqa
    from past.builtins import basestring  # noqa
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


from raredecay.tools import dev_tool


def _make_data(original_data, target_data=None, features=None, target_from_data=False,
               weights_ratio=0, weights_original=None, weights_target=None):
    """Return the concatenated data, weights and labels for classifier training.

     Differs to only *make_dataset* from the |hepds_type| by providing the
     possibility of using other weights.
    """
    # make temporary weights if specific weights are given as parameters
    temp_ori_weights = None
    temp_tar_weights = None
    if not dev_tool.is_in_primitive(weights_original, None):
        temp_ori_weights = original_data.weights
        original_data.set_weights(weights_original)
    if not dev_tool.is_in_primitive(weights_target, None):
        temp_tar_weights = target_data.weights
        target_data.set_weights(weights_target)

    # create the data, target and weights
    data_out = original_data.make_dataset(target_data, columns=features,
                                          targets_from_data=target_from_data,
                                          weights_ratio=weights_ratio)

    # reassign weights if specific weights have been used
    if not dev_tool.is_in_primitive(temp_ori_weights, None):
        original_data.set_weights(temp_ori_weights)
    if not dev_tool.is_in_primitive(temp_tar_weights, None):
        original_data.set_weights(temp_tar_weights)

    return data_out
