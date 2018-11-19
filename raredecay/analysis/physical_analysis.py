# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:49:45 2016

@author: Jonas Eschle "Mayou36"


DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat

DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!



This module provides high-level function, which often contain an essential
part of a complete MVA. The functions are sometimes quite verbous, both
in plotting as well as in printing, but always also return the important
values.
"""
# Python 2 backwards compatibility overhead START
from __future__ import absolute_import, division, print_function, unicode_literals

import sys  # noqa
import warnings  # noqa
from builtins import (int,  # noqa
                      range, str, zip,
                      )  # noqa

import raredecay.meta_config  # noqa
import raredecay.tools.ml_scores

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

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raredecay.tools import dev_tool
import raredecay.analysis.ml_analysis as ml_ana

# legacy
from raredecay.analysis.compatibility_reweight import reweight


def _cut(data):
    from raredecay.tools import data_tools

    return data_tools.apply_cuts(*data)


# def preselection_cut(signal_data, bkg_data, percent_sig_to_keep=100):
#     """Cut the bkg while maintaining a certain percent of the signal. WIP."""
#
#     # import raredecay.meta_config as meta_cfg
#     from raredecay.tools import data_tools
#     from raredecay.globals_ import out
#     # from raredecay.tools.data_storage import HEPDataStorage
#
#     columns = signal_data.columns
#     signal_data.plot(figure="Before cut", title="Data comparison before cut")
#     signal_data.plot(figure="Signal comparison", title="Data comparison before cut vs after")
#     bkg_data.plot(figure="Background comparison", title="Data comparison before cut vs after")
#     bkg_data.plot(figure="Before cut")
#     bkg_length = len(bkg_data)
#     signal_length = len(signal_data)
#     signal_cp = signal_data.copy_storage()
#     bkg_cp = bkg_data.copy_storage()
#     signal_data = signal_data.pandasDF()
#     bkg_data = bkg_data.pandasDF()
#
#     applied_cuts = {}
#
#     percent_end = percent_sig_to_keep
#     percent_sig_to_keep = 100
#     stepsize = 0.1
#     keep = {}
#
#     while True:
#
#         #        pool = multiprocessing.Pool(meta_cfg.n_cpu_max)
#         sig = np.array([signal_data.as_matrix()[:, i] for i, _t in enumerate(columns)])
#         sig = copy.deepcopy(sig)
#         bkg = np.array([bkg_data.as_matrix()[:, i] for i, _t in enumerate(columns)])
#         bkg = copy.deepcopy(bkg)
#         data = list(zip(sig, bkg, [percent_sig_to_keep] * len(columns)))
#         limits, rejection = [], []
#         for sig, bkg, per in data:
#             temp = data_tools.apply_cuts(sig, bkg, per, bkg_length=bkg_length)
#             limits.append(temp[0])
#             rejection.append(temp[1])
#         # limits, rejection = pool.map(_cut, data)
#         i_max_rej = np.argmax(rejection)
#         max_rejection = np.max(rejection)
#         column, limits = columns[i_max_rej], limits[i_max_rej]
#         print(percent_sig_to_keep, percent_end)
#         if max_rejection < 0.001 and percent_sig_to_keep == 100:
#             if percent_end < 100:
#                 percent_sig_to_keep -= stepsize
#             else:
#                 break
#         elif percent_end <= percent_sig_to_keep < 100:
#             percent_end += stepsize
#             stepsize *= (100 - stepsize) / 100
#         elif percent_sig_to_keep < percent_end:
#             break
#
#         if column in applied_cuts:
#             max_rejection += applied_cuts[column]['reduction']
#         applied_cuts[column] = {"limits": limits, "reduction": max_rejection}
#
#         cuts = np.logical_and(signal_data[column] > limits[0], signal_data[column] < limits[1])
#         signal_data = signal_data[cuts]
#
#         cuts = np.logical_and(bkg_data[column] > limits[0], bkg_data[column] < limits[1])
#         bkg_data = bkg_data[cuts]
#         print("We used " + column)
#
#     # signal_data.hist(bins=30)
#     #    bkg_data.hist(bins=30)
#
#     signal_len_cut = len(np.array(signal_data.as_matrix()[:, 0]))
#     bkg_len_cut = len(np.array(bkg_data.as_matrix()[:, 0]))
#     signal_cp.set_data(signal_data)
#     signal_cp.plot(figure="Signal comparison")
#     signal_cp.plot(figure="Data cut plt", title="Data with cuts applied", log_y_axes=True)
#
#     bkg_cp.set_data(bkg_data)
#     bkg_cp.plot(figure="Background comparison")
#     bkg_cp.plot(figure="Data cut plt", log_y_axes=True)
#
#     out.add_output(applied_cuts, section="Preselection cuts report")
#     out.add_output(keep, section="All limits")
#     bkg_rejection = sum([i['reduction'] for i in list(applied_cuts.values())])
#     out.add_output(["summed up Bkg rejection: ", bkg_rejection, "True rejection: ",
#                     100.0 - (bkg_len_cut / bkg_length), " True remaining signal: ",
#                     signal_len_cut / signal_length], section="Total bkg rejection")
#     print(signal_len_cut)
#     print(signal_length)
#     print(bkg_len_cut)
#     print(bkg_length)
#
#     return applied_cuts


def feature_exploration(original_data, target_data, features=None, n_folds=10,
                        clf='xgb', roc_auc='single', extended_report=True):
    """Explore the features by getting the roc auc and their feature importance.

    An essential part is to have a rough idea of how discriminating the
    features are. A classifier is trained on each single feature and all
    together, correlations and feature importance are plottet if wanted.

    Parameters
    ----------
    original_data : |hepds_type|
        One dataset
    target_data : |hepds_type|
        The other dataset
    features : list(str, str, str,...)
        The features/branches/columns to explore
    n_folds : int > 1
        Number of folds to split the data into to do some training/testing and
        get an estimate for the feature importance.
    clf : str or dict or clf, see: :py:meth:`~raredecay.analysis.ml_analysis.make_clf()`
        The classifier you want to use.
    roc_auc : {'single', 'all', 'both'} or False
        Whether to make a training/testing with:

        - every single feature (-> n_feature times KFolded training)
        - all features together (-> one KFolded training)
        - both of the above
        - None of them (-> use *False*)
    extended_report : boolean
        If True, an extended report will be made including feature importance
        and more.

    """
    features = dev_tool.entries_to_str(features)
    clf = dev_tool.entries_to_str(clf)
    roc_auc = dev_tool.entries_to_str(roc_auc)

    roc_auc_all = True if roc_auc in ('all', 'both') else False
    roc_auc_single = True if roc_auc in ('single', 'both') else False

    if features is not None:
        original_data = original_data.copy_storage(columns=features)
        target_data = target_data.copy_storage(columns=features)

    figure = "Plotting" + str(original_data.name) + " and " + str(target_data.name)
    if extended_report:
        original_data.plot(figure=figure, title=figure)
        target_data.plot(figure=figure)

    if roc_auc_all:
        ml_ana.classify(original_data, target_data, validation=n_folds,
                        extended_report=extended_report, clf=clf,
                        curve_name="all features", plot_title="ROC AUC of")

    features = original_data.columns if features is None else features

    output = {}
    out_temp = {}
    if roc_auc_single:
        for feature in features:
            title = "Feature exploration, ROC AUC only using " + str(feature)
            tmp_, score = ml_ana.classify(original_data, target_data, features=feature,
                                          curve_name="only using " + str(feature), clf=clf,
                                          validation=n_folds, extended_report=extended_report,
                                          plot_title=title, weights_ratio=1, plot_importance=2)
            del tmp_
            out_temp[feature] = score

    output['score'] = out_temp

    return output


def final_training(real_data, mc_data, bkg_sel, clf='xgb', n_folds=10, columns=None,
                   performance_only=True, metric_vs_cut='punzi', weights_ratio=0,
                   save_real_pred=False, save_mc_pred=False):
    """Train on bkg and MC, test metric, performance and predict probabilities.

    The goal of an MVA is to have certain probabilities predicted for each
    event to make further cuts on the data-sample and reduce the background.

    There are two modes to run:
        - **performance_only**: train a clf K-folded on the background and the
          MC and predict, then create the ROC-curve and plot a metric. This
          is to get an idea of how well the classifier performs as well as
          to find the optimal cutoff-value on the predictions.
        - **prediction_mode**: (*set performance_only to False*) train a clf
          on the bkg and MC and predict K-folded the probabilities for all
          data (bkg, MC and the rest) without any event occuring in the
          training-set as well as in the test-set. If a name is given to
          *save_mc_pred* respectively *save_real_pred*, the predicitions will
          be saved to the root-file the data was taken from.

    Parameters
    ----------
    real_data : |hepds_type|
        The real data
    mc_data : |hepds_type|
        The MC data (signal)
    bkg_sel : str or [str]
        A string pointing to a column in the root-tree which tells if an event
        belongs to the bkg (1) to train on or not (0). This typically is
        something like this: (B_M > 5700) or similar
    clf : str or clf or dict, see :py:func:`~raredecay.analysis.ml_analysis.make_clf()`
        The classifier to be used.
    n_folds : int > 1
        The number of folds to use for the training
    columns : list(str, str, str,...)
        The columns to train on
    performance_only : boolean
        If True, the function is run in performance mode and does not predict
        but only creates a ROC-curve and a metric-vs-cut.
    metric_vs_cut : str {'punzi', 'precision'}
        The metric to test on the predictions.
    save_real_pred : str or False
        If provided, the predictions of the real data will be saved to its
        root-tree with the branch name specified here.
    save_mc_pred : str or False
        If provided, the predictions of the MC will be saved to its
        root-tree with the branch name specified here.

    Return
    ------
    out : dict
        Return a dict containing the following keys and values:

        - *best_threshold_cut*: The best cut for the given metric (only if metric_vs_cut is given)
        - *best_metric*: The highest value of the metric, basically "metric(best_threshold_cut)" (only if
            metric_vs_cut is given)
        - *pred_real*: The predicitions of real_data
        - *pred_mc*: The predicitions of mc_data

    """
    from raredecay.globals_ import out
    from raredecay.analysis.ml_analysis import classify
    from raredecay.tools.metrics import punzi_fom, precision_measure

    # Python 2/3 compatibility
    bkg_sel = dev_tool.entries_to_str(bkg_sel)
    clf = dev_tool.entries_to_str(clf)
    columns = dev_tool.entries_to_str(columns)
    metric_vs_cut = dev_tool.entries_to_str(metric_vs_cut)
    save_mc_pred = dev_tool.entries_to_str(save_mc_pred)
    save_real_pred = dev_tool.entries_to_str(save_real_pred)

    # check if predictions can be saved: need to be root-file and no selection applied
    if performance_only:
        save_real_pred = save_mc_pred = False

    output = {}
    if save_real_pred:
        if real_data.data_type != 'root':
            raise TypeError("Real predictions should be saved but data is not a root-file but " +
                            real_data.data_type)
        elif not (real_data.data['selection'] is None):
            raise ValueError("Real pred set to be saved, but has selection " +
                             real_data.data['selection'] + " applied")
    if save_mc_pred:
        if mc_data.data_type != 'root':
            raise TypeError("MC predictions should be saved but data is not a root-file but " +
                            mc_data.data_type)
        elif not (mc_data.data['selection'] is None):
            raise ValueError("MC pred set to be saved, but has selection " +
                             mc_data.data['selection'] + " applied")

    # backwards compatibility


    bkg_sel = [bkg_sel] if not isinstance(bkg_sel, (list, tuple)) else bkg_sel
    if bkg_sel[0].startswith('noexpand:'):
        bkg_sel = bkg_sel[0][9:]
    else:
        bkg_sel = bkg_sel[0]

    pred_real = []
    pred_mc = []

    predict = not performance_only
    if performance_only:
        bkg_df = real_data.pandasDF()
        bkg_df = bkg_df.ix[np.array(bkg_df[bkg_sel].T) == 1]
        bkg_data = real_data.copy_storage()
        bkg_data.set_data(bkg_df)
        del bkg_df
        _clf, kfold_score, pred_tmp = classify(bkg_data, mc_data, validation=n_folds, clf=clf,
                                               get_predictions=True, extended_report=True,
                                               features=columns, weights_ratio=weights_ratio)

        report = pred_tmp['report']

        if metric_vs_cut == 'punzi':
            metric = punzi_fom
            title = "Punzi FoM vs threshold cut on " + real_data.name
        elif metric_vs_cut == 'precision':
            metric = precision_measure
            metric.__name__ = r"precision $\frac {n_{signal}} {\sqrt{n_{signal} + n_{background}}}$"
            title = "Precision vs threshold cut on " + real_data.name
        elif metric_vs_cut:
            raise ValueError("Invalid metric: " + str(metric_vs_cut))

        if metric_vs_cut:
            out.figure(title)
            # plt.legend()
            from rep.report.metrics import OptimalMetric
            metric_optimal = OptimalMetric(metric,
                                           expected_s=sum(mc_data.get_weights(normalize=False)),
                                           expected_b=sum(bkg_data.get_weights(normalize=False))
                                           )
            metric_optimal.plot_vs_cut(y_true=pred_tmp['y_true'],
                                       proba=pred_tmp['y_proba'],
                                       sample_weight=pred_tmp['weights']).plot(fontsize=25)
            best_cut, best_metric = metric_optimal.compute(y_true=pred_tmp['y_true'],
                                                           proba=pred_tmp['y_proba'],
                                                           sample_weight=pred_tmp['weights'])
            best_index = np.argmax(best_metric)
            output = {'best_threshold_cut': best_cut[best_index],
                      'best_metric': best_metric[best_index]
                      }

    # predict to all data
    if predict:

        # make folds and loop through
        real_data.make_folds(n_folds=n_folds)
        mc_data.make_folds(n_folds=n_folds)
        for i in range(n_folds):
            real_train, real_test = real_data.get_fold(i)
            mc_train, mc_test = mc_data.get_fold(i)
            real_train.data_name_addition = "train"
            real_test.data_name_addition = "test"
            bkg_df = real_train.pandasDF()

            bkg_df = bkg_df.ix[np.array(bkg_df[bkg_sel].T) == 1]
            real_train.set_data(bkg_df)
            real_train.data_name_addition = "train bkg"

            real_test_index = real_test.index
            mc_test_index = mc_test.index
            first_example = i == 0  # to plot only the first time
            clf_trained, _, pred_real_tmp = classify(real_train, mc_train, validation=real_test,
                                                     clf=clf, get_predictions=True,
                                                     extended_report=first_example,
                                                     features=columns, weights_ratio=weights_ratio)
            clf_trained, _, pred_mc_tmp = classify(validation=mc_test, clf=clf_trained,
                                                   get_predictions=True,
                                                   extended_report=first_example,
                                                   features=columns, weights_ratio=weights_ratio)

            # collect predictions and index
            pred_real_tmp = pred_real_tmp['y_proba']
            pred_real.append(pd.Series(pred_real_tmp[:, 1], index=real_test_index))
            pred_mc_tmp = pred_mc_tmp['y_proba']
            pred_mc.append(pd.Series(pred_mc_tmp[:, 1], index=mc_test_index))

        # concatenate predictions and index
        pred_real = pd.concat(pred_real)
        pred_real.sort_index(inplace=True)
        pred_mc = pd.concat(pred_mc)
        pred_mc.sort_index(inplace=True)
        output['pred_real'] = pred_real
        output['pred_mc'] = pred_mc

        # save predictions
        if isinstance(save_real_pred, (basestring, int)) and not isinstance(save_real_pred, bool) and predict:
            root_dict = copy.deepcopy(real_data.data)

            if root_dict['selection'] is not None:
                raise ValueError(
                        "Cannot save predictions to root as selections have been applied in the script")

            add_branch_to_rootfile(filename=root_dict['filenames'],
                                   treename=root_dict['treename'],
                                   new_branch=pred_real, branch_name=save_real_pred)

        if isinstance(save_mc_pred, (basestring, int)) and not isinstance(save_mc_pred, bool) and predict:
            root_dict = copy.deepcopy(mc_data.data)

            if root_dict['selection'] is not None:
                raise ValueError(
                        "Cannot save predictions to root as selections have been applied in the script")

            add_branch_to_rootfile(filename=root_dict['filenames'],
                                   treename=root_dict.get('treename'),
                                   new_branch=pred_mc, branch_name=save_mc_pred)
        out.figure("predictions total")
        plt.legend()
        plt.title("Predictions of MC vs all real data")
        plt.hist(pred_real, bins=30)
        plt.hist(pred_mc, bins=30)

        out.figure("predictions total normalized")
        plt.legend()
        plt.title("Predictions of MC vs all real data normalized")
        plt.hist(pred_real, bins=30, normed=True, alpha=0.5, range=(0, 1))
        plt.hist(pred_mc, bins=30, normed=True, alpha=0.5, range=(0, 1))

    return output


def add_branch_to_rootfile(filename, treename, new_branch, branch_name,
                           overwrite=True):
    """Add a branch to a given ROOT-Tree.

    Add some data (*new_branch*) to the ROOT-file (*filename*) into its tree
    (*treename*) under the branch (*branch_name*)

    Parameters
    ----------
    filename : str
        The name of the file (and its path)
    treename : str
        The name of the tree to save the data in
    new_branch : array-like
        The data to add to the root-file
    branch_name : str
        The name of the branch the data will be written too. This can either be
        a new one or an already existing one, which then will be overwritten.
        No "friend" will be created.
    overwrite : boolean
        NOT IMPLEMENTED!
    """
    from raredecay.tools import data_tools
    from raredecay.globals_ import out

    root_data = {'filenames': filename, 'treename': treename}
    status = data_tools.add_to_rootfile(root_data, new_branch=new_branch,
                                        branch_name=branch_name, overwrite=overwrite)
    if status == 0:
        out.add_output(["Added succesfully", new_branch, "as", branch_name, "to",
                        filename], obj_separator=" ")
    elif status == 1:
        out.add_output(["Did not add", new_branch, "as", branch_name, "to",
                        filename, "because it already exists and overwrite is set to false"],
                       obj_separator=" ")


# OLD


def reweightCV(real_data, mc_data, columns=None, n_folds=10,
               reweighter='gb', reweight_cfg=None, n_reweights=1,
               scoring=True, score_columns=None, n_folds_scoring=10, score_clf='xgb',
               mayou_score=False, extended_train_similar=False, apply_weights=True):
    """Reweight data MC/real in a K-Fold way to unbias the reweighting.

    Sophisticated reweighting-algorithms can be quite sensitive to its
    hyperparameters. Therefore, it is good to get an estimation for the
    reweighting quality by reweighting the data itself and "test" it (compare how
    similar the reweighted to the real one is). In order to get an unbiased
    reweighting, a KFolding procedure is applied:

    - the reweighter is trained on n-1/nth of the data and predicts the weights for the 1/n leftover.
    - This is done n times resulting in unbiased weights for the mc data.

    To know, how well the reweighter worked, different stategies can be used
    and are implemented, for further information also see: |reweightingCV_quality_measure_link|

    Parameters
    ----------
    real_data : |hepds_type|
        The real data
    mc_data : |hepds_type|
        The mc data
    columns : list(str, str, str, ...)
        The branches to use for the reweighting.
    n_folds : int > 1
        Number of folds to split the data for the reweighting. Usually, the
        higher the better.
    reweighter : str {'gb', 'bins'}
        Which reweighter to use, either the Gradient Boosted reweighter or the
        (normally used) Bins reweighter (both from *hep_ml*)
    reweight_cfg : dict
        A dict containing all the keyword arguments for the configuration of
        the reweighters.
    n_reweights : int
        As the reweighting often yields different weights depending on random
        parameters like the splitting of the data, the new weights can be
        produced by taking the average of the weights over many reweighting
        runs. n_reweights is the number of reweight runs to average over.
    scoring : boolean
        If True, the data is not only reweighted with KFolding but also several
        scoring metrics are tested.

        - Data-ROC : The data (mc reweighted and real mixed) is split in
          KFolds, a classifier is then trained on the training fold and tested
          on the test-fold. This is done K times and the roc curve is
          evaluated. It is a good measure, basically, for how well two datasets
          can be distinguished *but* can be "overfitted". Having too high,
          single weights can lead to a roc curve significantly lower then 0.5
          and therefore only a good indication but not a single measure of
          quality for the reweighter hyper-parameter search.
        - mcreweighted_as_real : n-1/n part of the data is trained on the
          reweighter and the last 1/n part is then reweighted (as described
          above). We can train a classifier on the mc (not reweighted) as
          well as the real data (so a classifier which "distinguishes" between
          mc and real) and predict:

          - (not in training used) mc (not reweighted) and label it as if it
            were real data.
          - (not in training used) mc reweighted and label it as if it were
            real data.
          - (not in training used) real data and label it real.

          Then we look at the tpr (we cannot look at the ROC as we only inserted
          one class of labels; real) and therefore at "how many of the
          datapoints we inserted did the classifier predict as real?":

          The score for the real data should be the highest, the one for the
          mc not reweighted the lowest. The reweighted one should be somewhere
          in between (most probably). It is **not** the goal to maximise the
          tpr for the mc reweighted (by changing the reweighter hyper-parameters)
          as high, single weights (which occure when overfitting) will increase
          the tpr drastically.
        - train_similar: The probably most stable score to find the gbreweighter
          hyper-parameters. The data is split into KFolds and a classifier is
          trained on the mc reweighted and real data. Then it predicts the
          (not yet seen) real data. The more it is able to predict as real,
          the more it was able to learn from the differences of the datasets.
          This scoring cannot overfit the same way the one above because a
          single, high weight will cause a very bad distribution of the mc
          data and therefore the classifier will be able to predict nearly
          every real data as real (only *one single point*, the one with
          the high weight, will be predicted as mc, the rest as real)
    score_columns : list(str, str, str,...)
        The columns to use for the scoring. They should not be the same as for
        the reweighting in order to unbias the score. It is usually a good
        idea to use the same branches as will be used for the selection
        training later on.
    n_folds_scoring : int > 1
        The number of folds to split the data into for the scoring
        described above.
    score_clf : str or dict or clf
        The classifier to use for the scoring. For an overview of what can be
        used, see :py:func:`~raredecay.analysis.ml_analysis.make_clf()`.
    mayou_score : boolean
        If True, the experimental *mayou_score* will be generated.
    extended_train_similar : boolean
        If True, an experimental score will be generated.
    apply_weights : boolean
        If True, set the new weights to the MC data in place. This changes the
        weights in the data-storage.


    Return
    ------
    out : dict
        The output is a dictionary containing the different scores and/or the
        new weights. The keywords are:

        - *weights* : pandas Series containing the new weights
        - *mcreweighted_as_real_score* : The scores of this method in a dict
        - *train_similar* : The scores of this method in a dict
        - *roc_auc_score* : The scores of this method in a dict
    """
    import numpy as np

    from raredecay.tools import metrics
    from raredecay.globals_ import out

    output = {}
    # do the Kfold reweighting. This reweights the data with Kfolding and returns
    # the weights. If add_weights_to_data is True, the weights will automatically be
    # added to the mc_data (or here, reweight_mc). To get an estimate
    # wheter it has over-fitted, you can get the mcreweighted_as_real_score.
    # This trains a clf on mc/real and tests it on mc, mc reweighted, real
    # but both labeled with the same target as the real data in training
    # The mc reweighted score should therefore lie in between the mc and the
    # real score.
    #    if not apply_weights:
    old_weights = mc_data.get_weights()
    # make sure the targets are set the right way TODO
    Kfold_output = ml_ana.reweight_Kfold(mc_data=mc_data, real_data=real_data,
                                         meta_cfg=reweight_cfg, columns=columns,
                                         reweighter=reweighter,
                                         n_reweights=n_reweights,
                                         mcreweighted_as_real_score=scoring,
                                         score_columns=score_columns,
                                         n_folds=n_folds, score_clf=score_clf,
                                         add_weights_to_data=True)
    new_weights = Kfold_output.pop('weights')
    # TODO: needed below?
    new_weights.sort_index()

    if scoring:
        output['mcreweighted_as_real_score'] = Kfold_output

        # To get a good estimation for the reweighting quality, the
        # train_similar score can be used. Its the one with training on
        # mc reweighted/real and test on real, quite robust.
        # Test_max is nice to know too even dough it can also be set to False if
        # testing the same distribution over and over again, as it is the same for
        # the same distributions (actually, it's just doing the score without the
        # weights).
        # test_predictions is an additional score I tried but so far I is not
        # reliable or understandable at all. The output, the scores dictionary,
        # is better described in the docs of the train_similar
        scores = raredecay.tools.ml_scores.train_similar_new(mc=mc_data, real=real_data, test_max=True,
                                                             n_folds=n_folds_scoring,
                                                             n_checks=n_folds_scoring,
                                                             columns=score_columns,
                                                             old_mc_weights=old_weights,
                                                             clf=score_clf)

        # scores = metrics.train_similar(mc_data=mc_data, real_data=real_data, test_max=True,
        #                                n_folds=n_folds_scoring, n_checks=n_folds_scoring,
        #                                features=score_columns, old_mc_weights=old_weights,
        #                                test_mc=extended_train_similar,
        #                                test_shuffle=extended_train_similar,
        #                                test_predictions=False, clf=score_clf)
        out.add_output(['Mayou FoM:', scores], to_end=True)

        # We can of course also test the normal ROC curve. This is weak to overfitting
        # but anyway (if not overfitting) a nice measure. You insert two datasets
        # and do the normal cross-validation on it. It's quite a multi-purpose
        # function depending on what validation is. If it is an integer, it means:
        # do cross-validation with n(=validation) folds.
        temp_mc_targets = mc_data.get_targets()
        mc_data.set_targets(0)
        temp_real_targets = real_data.get_targets()
        real_data.set_targets(1)
        tmp_, roc_auc_score, output = ml_ana.classify(original_data=mc_data, target_data=real_data,
                                                      validation=n_folds_scoring, plot_importance=4,
                                                      plot_title="ROC AUC to distinguish data",
                                                      clf=score_clf, weights_ratio=1,
                                                      features=score_columns,
                                                      extended_report=scoring,
                                                      get_predictions=True)
        del tmp_
        # HACK
        predictions = output['y_proba'][:, 1][output['y_true'] == 0]
        weights_pred = np.log(output['weights'][output['y_true'] == 0])
        weights_pred = output['weights'][output['y_true'] == 0]
        out.figure("Correlation of weights and predictions")
        plt.scatter(predictions, weights_pred)
        plt.xlabel("predictions")
        plt.ylabel("weights")
        out.figure("Correlation of weights and predictions hexbin")
        plt.hexbin(x=predictions, y=weights_pred, gridsize=150)
        #        sns.jointplot(x=predictions, y=weights_pred, kind="hex")



        if mayou_score:
            raredecay.tools.ml_scores.mayou_score(mc_data=mc_data, real_data=real_data,
                                                  n_folds=n_folds_scoring,
                                                  clf=score_clf, old_mc_weights=old_weights)
            # an example to add output with the most importand parameters. The first
            # one can also be a single object instead of a list. do_print means
            # printing it also to the console instead of only saving it to the output
            # file. To_end is sometimes quite useful, as it prints (and saves) the
            # arguments at the end of the file. So the important results are possibly
            # printed to the end
        out.add_output(['ROC AUC score:', roc_auc_score], importance=5,
                       title='ROC AUC of mc reweighted/real KFold', to_end=True)
        # TODO? NEW SCORES?
        #
        # out.add_output(['score:', scores['score'], "+-", scores['score_std']],
        #                importance=5,
        #                title='Train similar report', to_end=True)
        # if extended_train_similar:
        #     out.add_output(['\nScore_mc:', scores['score_mc'], "+-", scores['score_mc_std']],
        #                    importance=5,
        #                    to_end=True)
        # if scores.get('score_max', False):
        #     out.add_output(['score max:', scores['score_max'], "+-", scores['score_max_std']],
        #                    importance=5, to_end=True)
        # if scores.get('score_mc_max', False):
        #     out.add_output(['score_mc_max:', scores['score_mc_max'], "+-",
        #                     scores['score_mc_max_std']],
        #                    importance=5, to_end=True)

        if scores.get('score_shuffled', False):
            out.add_output(['score_shuffled:', scores['score_shuffled'], "+-",
                            scores['score_shuffled_std']],
                           importance=5, to_end=True)
        output['train_similar'] = scores
        output['roc_auc'] = roc_auc_score

    output['weights'] = new_weights
    if not apply_weights:
        mc_data.set_weights(old_weights)

    if scoring:
        mc_data.set_targets(temp_mc_targets)
        real_data.set_targets(temp_real_targets)

    return output

# temporary:
# if __name__ == '__main__':
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     from raredecay.tools.data_storage import HEPDataStorage
#
#     n_cols = 7
#     cols = [str(i) for i in range(n_cols)]
#     a = pd.DataFrame(np.random.normal(loc=0, scale=1, size=(1000, n_cols)), columns=cols)
#     a = HEPDataStorage(a, target=1, )
#     b = pd.DataFrame(np.random.normal(loc=0.2, scale=1.3, size=(1000, n_cols)), columns=cols)
#     b = HEPDataStorage(b, target=0)
#
#     #    feature_exploration(a, b, n_folds=3, roc_auc='all')
#
#
#
#     plt.show()
