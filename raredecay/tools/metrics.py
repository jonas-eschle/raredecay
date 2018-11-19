# -*- coding: utf-8 -*-
"""

@author: Jonas Eschle "Mayou36"

DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat

DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!


"""
# Python 2 backwards compatibility overhead START
from __future__ import absolute_import, division, print_function, unicode_literals  # noqa

import sys  # noqa
import warnings  # noqa
from builtins import (  # noqa
    str
    )  # noqa

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

import math as mt
import numpy as np

try:
    from raredecay.tools.ml_scores import (mayou_score, train_similar, train_similar_new, similar_dist,
                                       estimate_weights_bias,
                                       )
except ImportError:
    pass


# OLD


# NEW


def punzi_fom(n_signal, n_background, n_sigma=5):
    """Return the Punzi Figure of Merit = :math:`\\frac{S}{\sqrt(B) + n_{\sigma}/2}`.

    The Punzi FoM is mostly used for the detection of rare decays to prevent
    the metric of cutting off all the background and leaving us with only a
    very few signals.

    Parameters
    ----------
    n_signal : int or numpy.array
        Number of signals observed (= tpr; true positiv rate)
    n_background : int or numpy.array
        Number of background observed as signal (= fpr; false positiv rate)
    n_sigma : int or float
        The number of sigmas
    """  # pylint:disable=anomalous-backslash-in-string
    #     not necessary below??
    length = 1 if not hasattr(n_signal, "__len__") else len(n_signal)
    if length > 1:
        sqrt = np.sqrt(np.array(n_background))
        term1 = np.full(length, n_sigma / 2)
    else:

        sqrt = mt.sqrt(n_background)
        term1 = n_sigma / 2
    output = n_signal / (sqrt + term1)
    return output


def precision_measure(n_signal, n_background):
    """Return the precision measure = :math:`\\frac {n_{signal}} {\sqrt{n_{signal} + n_{background}}}`.

    Parameters
    ----------
    n_signal : int or numpy.array
        Number of signals observed (= tpr; true positiv rate)
    n_background : int or numpy.array
        Number of background observed as signal (= fpr; false positiv rate)
    n_sigma : int or float
        The number of sigmas

    """  # pylint:disable=anomalous-backslash-in-string
    try:
        length = len(n_signal)
    except TypeError:  # not an iterable
        length = 1
    if length > 1:
        sqrt = np.sqrt(np.array(n_signal + n_background))
        # HACK
        if sqrt[0] == 0:
            sqrt[0] = sqrt[1]
    else:
        sqrt = mt.sqrt(n_signal + n_background)
    output = n_signal / sqrt
    return output
