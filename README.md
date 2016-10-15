# raredecay #

This package consists of several tools for the event selection of particle decays, mostly built on machine learning techniques.
It contains:

- a **data-container** holding data, weights, labels and more and implemented root-to-python data conversion as well as plots and KFold-data splitting
- **reweighting** tools from the hep_ml-repository wrapped in a KFolding structure and with metrics to evaluate the reweighting quality
- **classifier optimization** tools for hyper-parameters as well as feature selection involving a backward-elimination
- an **output handler** which makes it easy to add text as well as figures into your code and automatically save them to a file
- ... and more


# Read the [Documentation](http://mayou36.bitbucket.org/index.html) #

*Still under work, like the rest of the README!*

# Or look at some [HowTo Examples](http://mayou36.bitbucket.org/index_howto.html) #






[pandas.DataFrame]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[LabeledDataStorage]: http://yandex.github.io/rep/data.html#module-rep.data.storage
[numpy.array]: http://docs.scipy.org/doc/numpy-1.10.1/user/basics.rec.html
[rootTree]: https://root.cern.ch/doc/v606/classTTree.html
