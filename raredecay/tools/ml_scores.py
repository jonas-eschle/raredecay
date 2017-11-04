# Python 2 backwards compatibility overhead START
"""
DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat

DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!

"""
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


import math as mt

import numpy as np

from raredecay.tools import dev_tool, data_storage


def mayou_score(mc_data, real_data, features=None, old_mc_weights=1,
                clf='xgb', splits=2, n_folds=10):
    """An experimental score using a "loss" function for data-similarity"""
    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.globals_ import out

    features = dev_tool.entries_to_str(features)
    clf = dev_tool.entries_to_str(clf)

    # initialize variables
    output = {}
    score_mc_vs_mcr = []
    score_mcr_vs_real = []
    #    splits *= 2  # because every split is done with fold 0 and 1 (<- 2 *)

    # loop over number of splits, split the mc data

    mc_data.make_folds(n_folds)
    real_data.make_folds(n_folds)

    # mc reweighted vs mc
    for fold in range(n_folds):
        mc_data_train, mc_data_test = mc_data.get_fold(fold)
        # TODO: no real folds? It is better to test on full data always?
        #        mc_data_train, mc_data_test = real_data.get_fold(fold)
        for split in range(splits * 2):  # because two possibilities per split
            if split % 2 == 0:
                mc_data_train.make_folds(2)
            mc_normal, mc_reweighted = mc_data_train.get_fold(split % 2)
            mc_normal.set_weights(old_mc_weights)
            score_mc_vs_mcr.append(ml_ana.classify(original_data=mc_normal,
                                                   target_data=mc_reweighted,
                                                   features=features,
                                                   validation=[mc_data_test, real_data],
                                                   clf=clf, plot_importance=1,
                                                   # TODO: no weights ratio? (roc auc)
                                                   weights_ratio=0
                                                   )[1])
    out.add_output(["mayou_score mc vs mc reweighted test on mc vs real score: ",
                    score_mc_vs_mcr, "\nMean: ", np.mean(score_mc_vs_mcr),
                    " +-", np.std(score_mc_vs_mcr) / mt.sqrt(len(score_mc_vs_mcr) - 1)],
                   subtitle="Mayou score", to_end=True)

    output['mc_distance'] = np.mean(score_mc_vs_mcr)

    # mc_reweighted vs real
    for fold in range(n_folds):
        real_train, real_test = real_data.get_fold(fold)
        mc_train, mc_test = mc_data.get_fold(fold)
        mc_test.set_weights(old_mc_weights)
        score_mcr_vs_real.append(ml_ana.classify(original_data=mc_train,
                                                 target_data=real_train,
                                                 features=features,
                                                 validation=[mc_test, real_test],
                                                 clf=clf, plot_importance=1,
                                                 # TODO: no weights ratio? (roc auc)
                                                 weights_ratio=0
                                                 )[1])

    out.add_output(["mayou_score real vs mc reweighted test on mc vs real score: ",
                    score_mcr_vs_real, "\nMean: ", np.mean(score_mcr_vs_real),
                    " +-", np.std(score_mcr_vs_real) / mt.sqrt(len(score_mcr_vs_real) - 1)],
                   to_end=True)

    output['real_distance'] = np.mean(score_mcr_vs_real)


def train_similar(mc_data, real_data, features=None, n_checks=10, n_folds=10,
                  clf='xgb', test_max=True, test_shuffle=True, test_mc=False,
                  old_mc_weights=1, test_predictions=False, clf_pred='rdf'):
    """Score for reweighting. Train clf on mc reweighted/real, test on real; minimize score.

    Enter two datasets and evaluate the score described below. Return a
    dictionary containing the different scores. The test_predictions is
    another scoring, which is built upon the train_similar method.

    **Scoring method description**

    **Idea**:
    A clf is trained on the reweighted mc as well as on the real data of a
    certain decay. Therefore, the classifier learns to distinguish between
    Monte-Carlo data and real data. Then we let the classifier predict some
    real data (an unbiased test set) and see, how many he is able to classify
    as real events. The lower the score, the less differences he was able to
    learn from the train data therefore the more similar the train data
    therefore the better the reweighting.

    **Advandages**: It is quite difficult to cheat on this method. Most of all
    it is robust to single high-weight events (which mcreweighted_as_real is
    not) and, in general, seems to be the best scoring so far.

    **Disadvantages**: If you insert a gaussian shaped 1.0 as mc and a gaussian
    shaped 1.1 as real, the score will be badly (around 0.33). So far, this was
    only observed for "artificial" distributions (even dough, of course, we
    do not know if it affects real distributions aswell partly)

    **Output explanation**

    The return is a dictionary containing several values. Of course, only the
    values, which are set to be evaluated, are contained. The keys are:

    - '**score**' : The average of all train_similar scores (as we use KFolding,
      there will be n_folds scores). *The* score.
    - '**score_std**' : The std of a single score, just for curiosity
    - '**score_max**' : The (average of all) "maximum" score. Actually the
      train_similar score but
      with mc instead of *reweighted* mc. Should be higher then the
      reweighted score.
    - '**score_max_std**' : The std of a single score, just for curiosity
    - '**score_pred**' : The score of the test_predictions method.
    - '**score_mc_pred**' : The score of the test_predictions method but on the
      predictions of the mc instead of the *reweighted* mc.

    Parameters
    ----------
    mc_data : |hepds_type|
        The reweighted Monte-Carlo data, assuming the new weights are applied
        already.
    real_data : |hepds_type|
        The real data
    n_checks : int >= 1
        Number of checks to perform. Has to be <= n_folds
    n_folds : int > 1
        Number of folds the data will be split into
    clf : str
        The name of a classifier to be used in
        :py:func:`~raredecay.analysis.ml_analysis.classify`.
    test_max : boolean
        If true, test for the "maximum value" by training also on mc/real
        (instead of *reweighted* mc/real)
        and test on real. The score for only mc should be higher than for
        reweighted mc/real. It *should* most probably but does not have to
        be!
    old_mc_weights : array-like or 1
        If *test_max* is True, the weights for mc before reweighting will be
        taken to be *old_mc_weights*, the weights the mc distribution had
        before the reweighting. The default is 1.
    test_predictions : boolean
        If true, try to distinguish the predictions. Advanced feature and not
        yet really discoverd how to interpret. Gives very high ROC somehow.
    clf_pred : str
        The classifier to be used to distinguish the predictions. Required for
        the *test_predictions*.

    Return
    ------
    out : dict
        A dictionary conaining the different scores. Description see above.

    """
    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.globals_ import out

    features = dev_tool.entries_to_str(features)
    clf = dev_tool.entries_to_str(clf)
    clf_pred = dev_tool.entries_to_str(clf_pred)

    # initialize variables
    assert 1 <= n_checks <= n_folds and n_folds > 1, "wrong n_checks/n_folds. Check the docs"
    assert isinstance(mc_data, data_storage.HEPDataStorage), \
        "mc_data wrong type:" + str(type(mc_data)) + ", has to be HEPDataStorage"
    assert isinstance(real_data, data_storage.HEPDataStorage), \
        "real_data wrong type:" + str(type(real_data)) + ", has to be HEPDataStorage"
    #    assert isinstance(clf, str),\
    #        "clf has to be a string, the name of a valid classifier. Check the docs!"

    output = {}

    scores = np.ones(n_checks)
    scores_shuffled = np.ones(n_checks)
    scores_mc = np.ones(n_checks)
    scores_max = np.ones(n_checks)  # required due to output of loop
    scores_mc_max = np.ones(n_checks)
    #    scores_weighted = []
    scores_max_weighted = []
    probas_mc = []
    probas_reweighted = []
    weights_mc = []
    weights_reweighted = []

    real_pred = []
    real_test_index = []
    real_mc_pred = []

    # initialize data
    tmp_mc_targets = mc_data.get_targets()
    mc_data.set_targets(0)
    real_data.make_folds(n_folds=n_folds)
    if test_mc:
        mc_data.make_folds(n_folds=n_folds)
    for fold in range(n_checks):
        real_train, real_test = real_data.get_fold(fold)
        if test_mc:
            mc_train, mc_test = mc_data.get_fold(fold)
            mc_test.set_targets(0)
        else:
            mc_train = mc_data.copy_storage()
        mc_train.set_targets(0)

        real_test.set_targets(1)
        real_train.set_targets(1)

        tmp_out = ml_ana.classify(mc_train, real_train, validation=real_test, clf=clf,
                                  plot_title="train on mc reweighted/real, test on real",
                                  weights_ratio=1, get_predictions=True,
                                  features=features,
                                  plot_importance=1, importance=1)
        clf_trained, scores[fold], pred_reweighted = tmp_out

        tmp_weights = mc_train.get_weights()

        if test_shuffle:
            import copy
            shuffled_weights = copy.deepcopy(tmp_weights)
            shuffled_weights.reindex(np.random.permutation(shuffled_weights.index))
            mc_train.set_weights(shuffled_weights)
            tmp_out = ml_ana.classify(mc_train, real_train, validation=real_test, clf=clf,
                                      plot_title="train on mc reweighted/real, test on real",
                                      weights_ratio=1, get_predictions=True,
                                      features=features,
                                      plot_importance=1, importance=1)
            scores_shuffled[fold] = tmp_out[1]
            mc_train.set_weights(tmp_weights)

        if test_mc:
            clf_trained, scores_mc[fold] = ml_ana.classify(validation=mc_test,
                                                           clf=clf_trained,
                                                           plot_title="train on mc reweighted/real, test on mc",
                                                           weights_ratio=1, get_predictions=False,
                                                           features=features,
                                                           plot_importance=1,
                                                           importance=1)

        # del clf_trained, tmp_pred
        probas_reweighted.append(pred_reweighted['y_proba'])
        weights_reweighted.append(pred_reweighted['weights'])

        real_pred.extend(pred_reweighted['y_pred'])
        real_test_index.extend(real_test.get_index())

        if test_max:
            temp_weights = mc_data.get_weights()
            mc_data.set_weights(old_mc_weights)
            tmp_out = ml_ana.classify(mc_data, real_train, validation=real_test,
                                      plot_title="real/mc NOT reweight trained, validate on real",
                                      weights_ratio=1, get_predictions=True, clf=clf,
                                      features=features,
                                      plot_importance=1, importance=1)
            clf_trained, scores_max[fold], pred_mc = tmp_out
            if test_mc:
                clf_trained, scores_mc_max[fold] = ml_ana.classify(validation=mc_test, clf=clf_trained,
                                                                   plot_title="train on mc NOT reweighted/real, test on mc",
                                                                   weights_ratio=1,
                                                                   get_predictions=False,
                                                                   features=features,
                                                                   plot_importance=1,
                                                                   importance=1)
            del clf_trained
            # HACK
            tmp_pred = pred_mc['y_proba'][:, 1] * pred_mc['weights']
            scores_max_weighted.extend(tmp_pred * (pred_mc['y_true'] * 2 - 1))

            # HACK END
            mc_data.set_weights(temp_weights)
            probas_mc.append(pred_mc['y_proba'])
            weights_mc.append(pred_mc['weights'])

            real_mc_pred.extend(pred_mc['y_pred'])

    output['score'] = np.round(scores.mean(), 4)
    output['score_std'] = np.round(scores.std(), 4)

    if test_shuffle:
        output['score_shuffled'] = np.round(scores_shuffled.mean(), 4)
        output['score_shuffled_std'] = np.round(scores_shuffled.std(), 4)

    if test_mc:
        output['score_mc'] = np.round(scores_mc.mean(), 4)
        output['score_mc_std'] = np.round(scores_mc.std(), 4)

    out.add_output(["Score train_similar (recall, lower means better): ",
                    str(output['score']) + " +- " + str(output['score_std'])],
                   subtitle="Clf trained on real/mc reweight, tested on real")
    if test_max:
        output['score_max'] = np.round(scores_max.mean(), 4)
        output['score_max_std'] = np.round(scores_max.std(), 4)
        if test_mc:
            output['score_mc_max'] = np.round(scores_mc_max.mean(), 4)
            output['score_mc_max_std'] = np.round(scores_mc_max.std(), 4)
        out.add_output(["No reweighting score: ", round(output['score_max'], 4)])

    if test_predictions:
        # test on the reweighted/real predictions
        real_data.set_targets(targets=real_pred, index=real_test_index)
        tmp_, score_pred = ml_ana.classify(real_data, target_from_data=True, clf=clf_pred,
                                           features=features,
                                           plot_title="train on predictions reweighted/real, real as target",
                                           weights_ratio=1, validation=n_checks, plot_importance=3)
        output['score_pred'] = round(score_pred, 4)

    if test_predictions and test_max:
        # test on the mc/real predictions
        real_data.set_targets(targets=real_mc_pred, index=real_test_index)
        tmp_, score_mc_pred = ml_ana.classify(real_data, target_from_data=True, clf=clf_pred,
                                              validation=n_checks,
                                              plot_title="mc not rew/real pred, real as target",
                                              weights_ratio=1, plot_importance=3)
        output['score_mc_pred'] = np.round(score_mc_pred, 4)

    mc_data.set_targets(tmp_mc_targets)

    output['similar_dist'] = similar_dist(predictions=np.concatenate(probas_reweighted)[:, 1],
                                          weights=np.concatenate(weights_reweighted))

    return output


def estimate_weights_bias(mc, real, columns=None, n_folds=10, clf='xgb'):
    pass


def train_similar_new(mc, real, columns=None, n_checks=10, n_folds=10, clf='xgb', test_max=True,
                      test_shuffle=True, test_mc=False, old_mc_weights=1, test_predictions=False,
                      clf_pred='rdf'):
    """Score for reweighting. Train clf on mc reweighted/real, test on real; minimize score.

    Enter two datasets and evaluate the score described below. Return a
    dictionary containing the different scores. The test_predictions is
    another scoring, which is built upon the train_similar method.

    **Scoring method description**

    **Idea**:
    A clf is trained on the reweighted mc as well as on the real data of a
    certain decay. Therefore, the classifier learns to distinguish between
    Monte-Carlo data and real data. Then we let the classifier predict some
    real data (an unbiased test set) and see, how many he is able to classify
    as real events. The lower the score, the less differences he was able to
    learn from the train data therefore the more similar the train data
    therefore the better the reweighting.

    **Advandages**: It is quite difficult to cheat on this method. Most of all
    it is robust to single high-weight events (which mcreweighted_as_real is
    not) and, in general, seems to be the best scoring so far.

    **Disadvantages**: If you insert a gaussian shaped 1.0 as mc and a gaussian
    shaped 1.1 as real, the score will be badly (around 0.33). So far, this was
    only observed for "artificial" distributions (even dough, of course, we
    do not know if it affects real distributions aswell partly)

    **Output explanation**

    The return is a dictionary containing several values. Of course, only the
    values, which are set to be evaluated, are contained. The keys are:

    - '**score**' : The average of all train_similar scores (as we use KFolding,
      there will be n_folds scores). *The* score.
    - '**score_std**' : The std of a single score, just for curiosity
    - '**score_max**' : The (average of all) "maximum" score. Actually the
      train_similar score but
      with mc instead of *reweighted* mc. Should be higher then the
      reweighted score.
    - '**score_max_std**' : The std of a single score, just for curiosity
    - '**score_pred**' : The score of the test_predictions method.
    - '**score_mc_pred**' : The score of the test_predictions method but on the
      predictions of the mc instead of the *reweighted* mc.

    Parameters
    ----------
    mc : |hepds_type|
        The reweighted Monte-Carlo data, assuming the new weights are applied
        already.
    real : |hepds_type|
        The real data
    n_checks : int >= 1
        Number of checks to perform. Has to be <= n_folds
    n_folds : int > 1
        Number of folds the data will be split into
    clf : str
        The name of a classifier to be used in
        :py:func:`~raredecay.analysis.ml_analysis.classify`.
    test_max : boolean
        If true, test for the "maximum value" by training also on mc/real
        (instead of *reweighted* mc/real)
        and test on real. The score for only mc should be higher than for
        reweighted mc/real. It *should* most probably but does not have to
        be!
    old_mc_weights : array-like or 1
        If *test_max* is True, the weights for mc before reweighting will be
        taken to be *old_mc_weights*, the weights the mc distribution had
        before the reweighting. The default is 1.
    test_predictions : boolean
        If true, try to distinguish the predictions. Advanced feature and not
        yet really discoverd how to interpret. Gives very high ROC somehow.
    clf_pred : str
        The classifier to be used to distinguish the predictions. Required for
        the *test_predictions*.

    Return
    ------
    out : dict
        A dictionary conaining the different scores. Description see above.

    """
    import raredecay.analysis.ml_analysis as ml_ana
    from raredecay.tools.data_storage import HEPDataStorage
    from raredecay.analysis import statistics

    # Python 2/3 compatibility, str
    columns = dev_tool.entries_to_str(columns)
    clf = dev_tool.entries_to_str(clf)
    clf_pred = dev_tool.entries_to_str(clf_pred)

    # initialize variables
    assert 1 <= n_checks <= n_folds and n_folds > 1, "wrong n_checks/n_folds. Check the docs"
    assert isinstance(mc, data_storage.HEPDataStorage), \
        "mc_data wrong type:" + str(type(mc)) + ", has to be HEPDataStorage"
    assert isinstance(real, data_storage.HEPDataStorage), \
        "real_data wrong type:" + str(type(real)) + ", has to be HEPDataStorage"
    #    assert isinstance(clf, str),\
    #        "clf has to be a string, the name of a valid classifier. Check the docs!"

    output = {}

    predictions = []
    predictions_weights = []
    predictions_max = []
    predictions_max_weights = []
    predictions_min = []
    predictions_min_weights = []

    # initialize data
    tmp_mc_targets = mc.get_targets()
    mc.set_targets(0)
    real.make_folds(n_folds=n_folds)

    for fold in range(n_checks):
        real_train, real_test = real.get_fold(fold)
        mc_train = mc.copy_storage()
        mc_train.set_targets(0)
        real_test.set_targets(1)
        real_train.set_targets(1)

        tmp_out = ml_ana.classify(mc_train, real_train, validation=real_test, clf=clf,
                                  plot_title="train on mc reweighted/real, test on real",
                                  weights_ratio=1, get_predictions=True,
                                  features=columns,
                                  plot_importance=1, importance=1)
        clf_trained, _, pred = tmp_out

        predictions.append(pred['y_proba'][:, 1])
        predictions_weights.append(pred['weights'])

        temp_weights = mc_train.weights
        mc_train.set_weights(old_mc_weights)
        tmp_out = ml_ana.classify(original_data=mc_train, target_data=real_train, validation=real_test,
                                  plot_title="real/mc NOT reweight trained, validate on real",
                                  weights_ratio=1, get_predictions=True, clf=clf,
                                  features=columns,
                                  plot_importance=1, importance=1)
        clf_trained, _, pred = tmp_out
        predictions_max.append(pred['y_proba'][:, 1])
        predictions_max_weights.append(pred['weights'])
        mc_train.set_weights(temp_weights)

    predictions = np.concatenate(predictions)
    predictions_weights = np.concatenate(predictions_weights)
    predictions_max = np.concatenate(predictions_max)
    predictions_max_weights = np.concatenate(predictions_max_weights)

    # mix mc and real to get a nice shape of two similar dists
    # TODO: commented below out
    mc.set_weights(old_mc_weights)
    mc.make_folds(2)
    real.make_folds(2)
    mc1, mc2 = mc.get_fold(0)
    real1, real2 = real.get_fold(0)

    data1, target1, weights1 = mc1.make_dataset(real1)
    data2, target2, weights2 = mc2.make_dataset(real2)

    data1 = HEPDataStorage(data=data1, sample_weights=weights1, target=0)
    data2 = HEPDataStorage(data=data2, sample_weights=weights2, target=1)

    tmp_out = ml_ana.classify(original_data=data1, target_data=data2, validation=n_folds,
                              plot_title="real/mc mixed",
                              weights_ratio=1, get_predictions=True, clf=clf,
                              features=columns,
                              plot_importance=1, importance=1)
    clf_trained, _, pred = tmp_out
    predictions_min = np.array(pred['y_proba'][:, 1])
    predictions_min_weights = np.array(pred['weights'])

    mc.set_weights(temp_weights)
    mc.set_targets(tmp_mc_targets)

    # HACK
    import matplotlib.pyplot as plt
    n_bins = 20
    plt.figure("comparing the predictions")
    plt.hist(predictions, alpha=0.3, label="predictions", bins=n_bins, normed=1)
    plt.hist(predictions_min, alpha=0.3, label="predictions_min", bins=n_bins, normed=1)
    plt.hist(predictions_max, alpha=0.3, label="predictions_max", bins=n_bins, normed=1)
    plt.legend()
    # plt.autoscale()

    output['similar_ks_minimize'] = statistics.ks_2samp(predictions, predictions_min,
                                                        weights1=predictions_weights,
                                                        weights2=predictions_min_weights)
    output['similar_ks_max'] = statistics.ks_2samp(predictions_max, predictions_min,
                                                   weights1=predictions_max_weights,
                                                   weights2=predictions_min_weights)
    output['similar_ks_maximize'] = statistics.ks_2samp(predictions, predictions_max,
                                                        weights1=predictions_weights,
                                                        weights2=predictions_max_weights)

    return output


def similar_dist(predictions, weights=None, true_y=1, threshold=0.5):
    """Metric to evaluate the predictions on one label only for similarity test.

    This metric is used inside the mayou_score

    Parameters
    ----------
    predictions : :py:class:`~np.array`
        The predicitons
    weights : array-like
        The weights for the predictions
    true_y : {0 , 1}
        The "true" label of the data
    threshold : float
        The threshold for the predictions to decide whether a point belongs
        to 0 or 1.
    """
    # HACK
    scale = 2  # otherwise, the predictions will be [-0.5, 0.5]
    # HACK END
    data_valid = min(predictions) < threshold < max(predictions)
    if not data_valid:
        raise ValueError("Predictions are all above or below the threshold")

    if true_y == 0:
        predictions = 1 - predictions

    predictions -= threshold
    predictions *= scale
    true_pred = predictions[predictions > 0]
    false_pred = predictions[predictions <= 0] * -1

    true_weights = false_weights = 1

    if not dev_tool.is_in_primitive(weights, None):
        true_weights = weights[predictions > 0]
        false_weights = weights[predictions <= 0]
    score = sum(((np.exp(1.3 * np.square(true_pred + 0.6)) - 1.5969) * 0.5) * true_weights)
    score -= sum(((np.sqrt(false_pred) - np.power(false_pred, 0.8)) * 2) * false_weights)
    score /= sum(weights)

    return score
