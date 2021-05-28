"""

@author: Jonas Eschle "Mayou36"

DEPRECEATED! USE OTHER MODULES LIKE rd.data, rd.ml, rd.reweight, rd.score and rd.stat

DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!DEPRECEATED!

"""


from raredecay.tools import dev_tool


def _make_data(
    original_data,
    target_data=None,
    features=None,
    target_from_data=False,
    weights_ratio=0,
    weights_original=None,
    weights_target=None,
):
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
    data_out = original_data.make_dataset(
        target_data,
        columns=features,
        targets_from_data=target_from_data,
        weights_ratio=weights_ratio,
    )

    # reassign weights if specific weights have been used
    if not dev_tool.is_in_primitive(temp_ori_weights, None):
        original_data.set_weights(temp_ori_weights)
    if not dev_tool.is_in_primitive(temp_tar_weights, None):
        original_data.set_weights(temp_tar_weights)

    return data_out
