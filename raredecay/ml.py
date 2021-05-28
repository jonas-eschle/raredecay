"""

@author: Jonas Eschle "Mayou36"

"""


try:
    from raredecay.analysis.ml_analysis import (
        classify,
        backward_feature_elimination,
        optimize_hyper_parameters,
        make_clf,
    )
    from raredecay.analysis.physical_analysis import feature_exploration
    from raredecay.analysis.physical_analysis import final_training as sideband_training

    __all__ = [
        "classify",
        "backward_feature_elimination",
        "optimize_hyper_parameters",
        "make_clf",
        "feature_exploration",
        "sideband_training",
    ]
except Exception as err:
    print("could not import machine learning algorithms (missing deps?)", str(err))
