# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: Jonas Eschle "Mayou36"

The reweighting module consists of function to reweight a distribution by learning from
two other distributions as well as reweighting a distribution by itself in a k-fold way.
"""
# Python 2 backwards compatibility overhead START
from __future__ import division, absolute_import, print_function, unicode_literals

import copy
import sys
import warnings

import hep_ml.reweight
import pandas as pd
from builtins import (int, str)

import raredecay.meta_config
from raredecay.globals_ import out
from raredecay.tools import dev_tool, data_tools

try:
    from future.builtins.disabled import (apply, cmp, coerce, execfile, file, long, raw_input,
                                          reduce, reload, unicode, xrange, StandardError)
    from future.standard_library import install_aliases

    install_aliases()
    from past.builtins import basestring
except ImportError as err:
    if sys.version_info[0] < 3:
        if raredecay.meta_config.SUPPRESS_FUTURE_IMPORT_ERROR:
            raredecay.meta_config.warning_occured()
            warnings.warn("Module future is not imported, error is suppressed. This means "
                          "Python 3 code is run under 2.7, which can cause unpredictable"
                          "errors. Best install the future package.", RuntimeWarning)
        else:
            raise err
    else:
        basestring = str
        # Python 2 backwards compatibility overhead END
# import configuration
import raredecay.meta_config as meta_cfg
import raredecay.config as cfg

# HACK as reweight also uses meta_cfg for reweight_cfg
meta_cfg_module = meta_cfg

logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


def reweight_train(mc, real, columns=None, reweighter='gb', reweight_cfg=None, reweight_saveas=None,
                   weights_ratio=1, weights_mc=None, weights_real=None):
    """Return a trained reweighter from a (mc/real) distribution comparison.

    | Reweighting a distribution is a "making them the same" by changing the
      weights of the bins (instead of 1) for each event. Mostly, and therefore
      the naming, you want to change the mc-distribution towards the real one.
    | There are two possibilities

    * normal bins reweighting:
       divides the bins from one distribution by the bins of the other
       distribution. Easy and fast, but unstable and inaccurat for higher
       dimensions.
    * Gradient Boosted reweighting:
       uses several decision trees to reweight the bins. Slower, but more
       accurat. Very useful in higher dimensions.
       But be aware, that you can easily screw up things by overfitting.

    Parameters
    ----------
    mc : |hepds_type|
        The Monte-Carlo data to compare with the real data.
    real : |hepds_type|
        Same as *mc_data* but for the real data.
    columns : list of strings
        The columns/features/branches you want to use for the reweighting.
    reweighter : {'gb', 'bins'}
        Specify which reweighter to be used.

        - **gb**: The GradientBoosted Reweighter from REP,
          :func:`~hep_ml.reweight.GBReweighter`
        - **bins**: The simple bins reweighter from REP,
          :func:`~hep_ml.reweight.BinsReweighter`
    reweight_saveas : string
        To save a trained reweighter in addition to return it. The value
        is the filepath + name.
    reweight_cfg : dict
        Contains the parameters for the bins/gb-reweighter. See also
        :func:`~hep_ml.reweight.BinsReweighter` and
        :func:`~hep_ml.reweight.GBReweighter`.
    weights_ratio : numeric or None, False
        The ratio of the sum of mc weights / sum of real weights. If set to
        one, the reweighter will learn from nicely normalized distributions.
        A value greater than 1 means there are in total more mc events
        than data points.
    weights_mc : numpy.array [n_samples]
        Explicit weights for the Monte-Carlo data. Only specify if you don't
        want to use the weights in the *HEPDataStorage*.
    weights_real : numpy.array [n_samples]
        Explicit weights for the real data. Only specify if you don't
        want to use the weights in the *HEPDataStorage*.

    Returns
    -------
    out : object of type reweighter
        Reweighter is trained to the data. Can, for example,
        be used with :func:`~hep_ml.reweight.GBReweighter.predict_weights`
    """
    __REWEIGHT_MODE = {'gb': 'GB', 'bins': 'Bins', 'bin': 'Bins'}

    # HACK
    from raredecay.analysis.ml_analysis import _make_data

    # Python 2/3 compatibility, str
    columns = dev_tool.entries_to_str(columns)
    reweighter = dev_tool.entries_to_str(reweighter)
    reweight_saveas = dev_tool.entries_to_str(reweight_saveas)
    reweight_cfg = dev_tool.entries_to_str(reweight_cfg)

    # check for valid user input
    if data_tools.is_pickle(reweighter):
        return data_tools.adv_return(reweighter, save_name=reweight_saveas)

    if reweighter not in __REWEIGHT_MODE:
        raise ValueError("Reweighter invalid: " + reweighter)

    reweighter = __REWEIGHT_MODE.get(reweighter.lower())
    reweighter += 'Reweighter'

    # logging and writing output
    msg = ["Reweighter:", reweighter, "with config:", reweight_cfg]
    logger.info(msg)

    out.add_output(msg + ["\nData used:\n", mc.name, " and ",
                          real.name, "\ncolumns used for the reweighter training:\n",
                          columns], section="Training the reweighter", obj_separator=" ")

    if columns is None:
        # use the intesection of both colomns
        common_cols = set(mc.columns)
        common_cols.intersection_update(real.columns)
        columns = list(common_cols)
        if columns != mc.columns or columns != real.columns:
            logger.warning("No columns specified for reweighting, took intersection" +
                           " of both dataset, as it's columns are not equal." +
                           "\nTherefore some columns were not used!")
            reweight_cfg.warning_occured()

    # create data
    normalize_real = 1 if weights_ratio else None
    mc, _t, mc_weights = _make_data(original_data=mc, features=columns,
                                    weights_original=weights_mc, weights_ratio=weights_ratio)
    real, _t, real_weights = _make_data(real, features=columns,
                                        weights_original=weights_real, weights_ratio=normalize_real)
    del _t

    # train the reweighter
    reweight_cfg = {} if reweight_cfg is None else reweight_cfg

    if reweighter == "GBReweighter":
        reweighter = hep_ml.reweight.GBReweighter(**reweight_cfg)
    elif reweighter == "BinsReweighter":
        reweighter = hep_ml.reweight.BinsReweighter(**reweight_cfg)
    reweighter.fit(original=mc, target=real,
                   original_weight=mc_weights, target_weight=real_weights)
    return data_tools.adv_return(reweighter, save_name=reweight_saveas)


def reweight_weights(apply_data, reweighter_trained, columns=None, normalize=True, add_weights=True):
    """Apply reweighter to the data and (add +) return the weights.

    Can be seen as a wrapper for the
    :py:func:`~hep_ml.reweight.GBReweighter.predict_weights` method.
    Additional functionality:
     * Takes a trained reweighter as argument, but can also unpickle one
       from a file.

    Parameters
    ----------
    apply_data : |hepds_type|
        The data for which the weights are predicted.
    reweighter_trained : (pickled) reweighter (*from hep_ml*)
        The trained reweighter, which predicts the new weights.
    columns : list(str, str, str,...)
        The columns to use for the reweighting.
    normalize : boolean or int
        If True, the weights will be normalized (scaled) to the value of
        normalize.
    add_weights : boolean
        If set to False, the weights will only be returned and not updated in
        the data (*HEPDataStorage*). If you want to use the data later on
        in the script with the new weights, set this value to True.

    Returns
    ------
    out : :py:class:`~pd.Series`
        Return an instance of pandas Series of shape [n_samples] containing the
        new weights.
    """
    # HACK
    from raredecay.analysis.ml_analysis import _make_data

    # Python 2/3 compatibility, str
    reweighter_trained = dev_tool.entries_to_str(reweighter_trained)
    columns = dev_tool.entries_to_str(columns)

    normalize = 1 if normalize is True else normalize

    reweighter_trained = data_tools.try_unpickle(reweighter_trained)
    if columns is None:
        columns = reweighter_trained.columns
    #    new_weights = reweighter_trained.predict_weights(reweight_data.pandasDF(),
    new_weights = reweighter_trained.predict_weights(apply_data.pandasDF(columns=columns),
                                                     original_weight=apply_data.weights)

    # write to output
    out.add_output(["Using the reweighter:\n", reweighter_trained, "\n to reweight ",
                    apply_data.name], obj_separator="")

    if isinstance(normalize, (int, float)) and not isinstance(normalize, bool):
        new_weights *= new_weights.size / new_weights.sum() * normalize

    new_weights = pd.Series(new_weights, index=apply_data.index)
    if add_weights:
        apply_data.set_weights(new_weights)
    return new_weights


# NEW
def reweight(apply_data=None, mc=None, real=None, columns=None, reweighter='gb', reweight_cfg=None,
             n_reweights=1,
             add_weights=True):
    """(Train a reweighter and) apply the reweighter to get new weights.

    Train a reweighter from the real data and the corresponding MC differences.
    Then, try to correct the apply data (MC as well) the same as the first
    MC would have been corrected to look like its real counterpart.

    Parameters
    ----------
    apply_data : |hepds_type|
        The data which shall be corrected
    real : |hepds_type|
        The real data to train the reweighter on
    mc : |hepds_type|
        The MC data to train the reweighter on
    columns : list(str, str, str,...)
        The branches to use for the reweighting process.
    reweighter : {'gb', 'bins'} or trained hep_ml-reweighter (also pickled)
        Either a string specifying which reweighter to use or an already
        trained reweighter from the hep_ml-package. The reweighter can also
        be a file-path (str) to a pickled reweighter.
    reweight_cfg : dict
        A dict containing all the keywords and values you want to specify as
        parameters to the reweighter.
    n_reweights : int
        To get more stable weights, the mean of each weight over many
        reweighting runs (training and predicting) can be used. The
        n_reweights specifies how many runs to do.
    add_weights : boolean
        If True, the weights will be added to the data directly, therefore
        the data-storage will be modified.

    Return
    ------
    out : dict
        Return a dict containing the weights as well as the reweighter.
        The keywords are:

        - *reweighter* : The trained reweighter
        - *weights* : pandas Series containing the new weights of the data.

    """
    import raredecay.analysis.ml_analysis as ml_ana

    #    from raredecay.globals_ import out
    from raredecay.tools import data_tools

    output = {}
    reweighter_list = False
    new_reweighter_list = []
    n_reweights = 1

    reweighter = data_tools.try_unpickle(reweighter)

    if isinstance(reweighter, list):
        n_reweights = len(reweighter)
        reweighter_list = copy.deepcopy(reweighter)

    for run in range(n_reweights):
        if reweighter_list:
            reweighter = reweighter_list[run]
        reweighter = data_tools.try_unpickle(reweighter)
        if reweighter in ('gb', 'bins'):
            new_reweighter = raredecay.analysis.reweight.reweight_train(mc=mc,
                                                                        real=real,
                                                                        columns=columns,
                                                                        reweight_cfg=reweight_cfg,
                                                                        reweighter=reweighter)
            # TODO: hack which adds columns, good idea?
            assert not hasattr(new_reweighter,
                               'columns'), "Newly created reweighter has column attribute, which should be set on the fly now. Changed object reweighter?"
            new_reweighter.columns = data_tools.to_list(columns)

        else:
            new_reweighter = reweighter

        if n_reweights > 1:
            new_reweighter_list.append(new_reweighter)
        else:
            new_reweighter_list = new_reweighter

        if apply_data is not None:
            tmp_weights = raredecay.analysis.reweight.reweight_weights(apply_data=apply_data,
                                                                       columns=columns,
                                                                       reweighter_trained=new_reweighter,
                                                                       add_weights=False)
            if run == 0:
                new_weights = tmp_weights
            else:
                new_weights += tmp_weights

    if apply_data is not None:
        new_weights /= n_reweights
        new_weights.sort_index()

        if add_weights:
            apply_data.set_weights(new_weights)
        output['weights'] = new_weights
    output['reweighter'] = new_reweighter_list

    return output
