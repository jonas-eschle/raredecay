"""
DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat


DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!

"""

import copy

import raredecay.analysis


def reweight(apply_data=None, real_data=None, mc_data=None, columns=None,
             reweighter='gb', reweight_cfg=None, n_reweights=1,
             apply_weights=True):
    """(Train a reweighter and) apply the reweighter to get new weights.

    Train a reweighter from the real data and the corresponding MC differences.
    Then, try to correct the apply data (MC as well) the same as the first
    MC would have been corrected to look like its real counterpart.

    Parameters
    ----------
    apply_data : |hepds_type|
        The data which shall be corrected
    real_data : |hepds_type|
        The real data to train the reweighter on
    mc_data : |hepds_type|
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
    apply_weights : boolean
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

    reweighter = data_tools.try_unpickle(reweighter)

    if isinstance(reweighter, list):
        n_reweights = len(reweighter)
        reweighter_list = copy.deepcopy(reweighter)

    for run in range(n_reweights):
        if reweighter_list:
            reweighter = reweighter_list[run]
        reweighter = data_tools.try_unpickle(reweighter)
        if reweighter in ('gb', 'bins'):
            new_reweighter = raredecay.analysis.reweight.reweight_train(mc=mc_data, real=real_data,
                                                                        columns=columns,
                                                                        reweighter=reweighter,
                                                                        reweight_cfg=reweight_cfg)
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

        if apply_data:
            tmp_weights = raredecay.analysis.reweight.reweight_weights(apply_data=apply_data,
                                                                       reweighter_trained=new_reweighter,
                                                                       columns=columns, add_weights=False)
            if run == 0:
                new_weights = tmp_weights
            else:
                new_weights += tmp_weights

    if apply_data:
        new_weights /= n_reweights
        # TODO: remove below?
        new_weights.sort_index()

        if apply_weights:
            apply_data.set_weights(new_weights)
        output['weights'] = new_weights
    output['reweighter'] = new_reweighter_list

    return output
