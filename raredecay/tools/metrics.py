"""

@author: Jonas Eschle "Mayou36"

DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat

DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!


"""


import math as mt

import numpy as np

try:
    from raredecay.tools.ml_scores import (
        estimate_weights_bias,
        mayou_score,
        similar_dist,
        train_similar,
        train_similar_new,
    )
except ImportError:
    pass


# OLD


# NEW


def punzi_fom(n_signal, n_background, n_sigma=5):
    """Return the Punzi Figure of Merit = :math:`\\frac{S}{\\sqrt(B) + n_{\\sigma}/2}`.

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
    """Return the precision measure = :math:`\\frac {n_{signal}} {\\sqrt{n_{signal} + n_{background}}}`.

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
