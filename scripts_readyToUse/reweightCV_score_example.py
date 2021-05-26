"""
Created on Wed Sep 21 17:31:56 2016

@author: Jonas Eschle "Mayou36"

Script to get good hyper-parameters for the GBReweighting by reweighting itself
with KFolds and test several scores.

The raredecay package is available at https://github.com/mayou36/raredecay
"""
from raredecay import settings

# first of all, it is advised to set some run-configurations like logger-level,
# verbosity, n_cpu and output-path. It is not mandatory to do this, but if you
# don't, the output won't be saved and the logger-level cannot be changed later.
# other things, like verbosity and n_cpu can be changed later on.
# Call this function BEFORE any other (raredecay-) package import
settings.initialize(
    output_path="/home/data/output",  # TODO: valid folder
    run_name="Test run",  # TODO: name of the folder
    run_message="This is a test run, hello world",  # TODO:
    prompt_for_input=True,  # TODO: will promt for input at the beginning and end
    n_cpu=-1,
)  # TODO: set n cpu

from raredecay.tools.data_storage import HEPDataStorage
from raredecay.analysis.physical_analysis import reweightCV, add_branch_to_rootfile
from raredecay.analysis.compatibility_reweight import reweight

# TODO: set the run mode
kfolded_reweighting = True  # If True, this is for hyper-parameter testing
# TODO: set the branches for the data to be loaded from the file
all_branches = [
    "B_PT",
    "nTracks",
    "nSPDHits",
    "B_FDCHI2_OWNPV",
    "B_DIRA_OWNPV",
    # 'B_IPCHI2_OWNPV', 'l1_PT', 'l1_IPCHI2_OWNPV','B_ENDVERTEX_CHI2',
    # 'h1_IPCHI2_OWNPV', 'h1_PT', 'h1_TRACK_TCHI2NDOF'
]


# ==============================================================================
# Create the data
# ==============================================================================
DATA_PATH = "/home/decay-data/"  # TODO: set your path to the data (or leave away)

# TODO: set your data
real_data_root = dict(
    filenames=DATA_PATH + "B2KpiLL-sWeights.root",
    treename="DecayTree",
    branches=all_branches,
)

# TODO: set the name and weights of your data
real_data = HEPDataStorage(
    data=real_data_root,
    sample_weights="signal_sw",  # takes the branch 'signal_sw' as weights
    data_name="Real data",
    data_name_addition="cut",
)
# TODO: same as above
mc_data = dict(
    filenames=DATA_PATH + "Bu2K1Jpsi-mm-Sim08g.root",
    treename="DecayTree",
    branches=all_branches,
)
mc_data = HEPDataStorage(data=mc_data, data_name="MC", data_name_addition="cut")

# TODO: same as above. Apply data is the MC which you want to be reweighted
apply_data = dict(
    filenames=DATA_PATH + "Bu2K1Jpsi-ee.root",
    treename="DecayTree",
    branches=all_branches,
)
apply_data = HEPDataStorage(data=mc_data, data_name="MC", data_name_addition="cut")

# The data will be plotted. Using the same figure will plot over each other
plot_branches = ["B_PT", "nTracks", "nSPDHits", "B_ENDVERTEX_CHI2"]
mc_data.plot(
    figure="Data comparison",
    title="Data comparison MC vs. real data",
    columns=plot_branches,
)
real_data.plot(figure="Data comparison", columns=plot_branches)

# ==============================================================================
# Specify the hyper-parameters of the reweighter
# ==============================================================================
# TODO: adjust the hyper-parameters
reweight_cfg = dict(  # GB reweighter configuration, comments are "good" values
    n_estimators=23,  # 25
    max_depth=3,  # 3-6 or number of features
    learning_rate=0.1,  # 0.1
    min_samples_leaf=200,  # 200
    loss_regularization=7,  # 3-8
    gb_args=dict(subsample=0.8, min_samples_split=200),  # 0.8  # 200
)

# TODO: Columns to use for the reweighting
reweight_columns = [
    "B_PT",
    # 'nTracks',
    "nSPDHits",
    "B_ENDVERTEX_CHI2",
    # 'B_eta'
]

# ==============================================================================
#  Call the reweighting function.
# ==============================================================================
if kfolded_reweighting:
    scores = reweightCV(
        real_data=real_data,
        mc_data=mc_data,
        n_folds=10,  # TODO: number of folds to split for the reweighting
        reweighter="gb",
        reweight_cfg=reweight_cfg,
        columns=reweight_columns,
        scoring=True,  # to calculate the quality measures
        n_folds_scoring=10,  # TODO: how many folds for the scoring metrics
        score_clf="xgb",  # can also be a dict containing {'xgb': config-dict}
        apply_weights=True,  # save the new weights to the mc_data
    )

    new_weights = scores.pop("weights")
    # print scores  # just for curiosity, the scores will be added anyway to the output

    # ==============================================================================
    # we could now add the weights to the root file (normaly, we dont want this to do,
    # but rather use the normal reweighting function to get new weights.
    # Remember, reweightCV is primarly to get the hyper-parameters, it is
    # self-reweighting which we norally don't want to do)
    # ==============================================================================


else:  # this is the "real" reweighting
    reweight_out = reweight(
        real_data=real_data,
        mc_data=mc_data,
        apply_data=apply_data,
        columns=reweight_columns,
        reweighter="gb",
        reweight_cfg=reweight_cfg,
        apply_weights=True,
    )
    new_weights = reweight_out["weights"]

    # TODO: change where to save the new weights and how to name it.
    # Will overwrite if branch exists
    add_branch_to_rootfile(
        filename=DATA_PATH + "Bu2K1Jpsi-ee-MC.root",
        treename="DecayTree",  # TODO: change name of tree to save
        new_branch=new_weights,
        branch_name="test1",  # TODO: change name of weights
    )

# at the end, we have to finalize our run (saving the output, if initialized
# with an output_path, and return it)
output = settings.finalize(
    show_plots=True, play_sound_at_end=False  # the easy way to show plots
)  # lets you realize it finished
