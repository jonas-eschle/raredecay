"""

@author: Jonas Eschle "Mayou36"

"""


try:
    from raredecay.analysis.ml_analysis import mcreweighted_as_real
    from raredecay.tools.ml_scores import mayou_score, train_similar, train_similar_new

    __all__ = [
        "train_similar_new",
        "train_similar",
        "mayou_score",
        "mcreweighted_as_real",
    ]

except Exception as err:
    print("could not import machine learning based scores (missing deps?)", str(err))
