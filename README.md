# README of the raredecay-package #


[**Documentation**](mayou36.bitbucket.org/index.html)

*Still under work, like the rest of the README!*




# Overview #
 The repository is created for a particle decay analysis. The idea was to create a workplace where you can focus on the data manipulation and analysis instead of things like difficult data-conversion, plotting, making good output and more. The core consists of:

- a solid data-structure, a wrapper around pandas DataFrame which provides plotting function, data-naming, creation of sub-folds, weight-, label-storage and more.  
- a machine learning analysis tool, which provides the most common used things for analysis like hyper-parameter optimization, classification etc. which works with the data-storage  
- an output-module, which takes care of writing log-files, creating folders, saving images and more.  
- the physical analysis module, where you can build your own analysis tools pretty easily.  
- configurations files which contain all the necessary things to use the implemented physical analysis modes.
  
### Data ###

- ROOT TTree: dictionary, containing all the keywords to access the right branches (more specific: the keyword arguments for the root2rec method from root_numpy)
- pandas Dataframe

## The core: _HEPDataStorage_ ##
 The idea is to create a data frame which perfectly fits the need for data analysis. It is not a general replacement for the [pandas DataFrame][pandas.DataFrame] or the [Labeled Data Storage][LabeledDataStorage] but a wrapper for the first one and an extension for the second one. Both and more data frames are accessible (or returned) by built-in methods. Therefore, using HEPDataStorage has no disadvantage.

[pandas.DataFrame]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[LabeledDataStorage]: http://yandex.github.io/rep/data.html#module-rep.data.storage
[numpy.array]: http://docs.scipy.org/doc/numpy-1.10.1/user/basics.rec.html
[rootTree]: https://root.cern.ch/doc/v606/classTTree.html