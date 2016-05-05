# README of the raredecay-package #

# What is it for? #
 The repository is created for a particle decay analysis. The core is a solid data-structure, a wrapper around pandas DataFrame and more. Furthermore, some useful methods are included which do the work. For example reweighting two datasets and train a classifier on it, to be able to see how much they differ.

## The core: _HEPDataStorage_ ##
 The idea is to create a data frame which perfectly fits the need for data analysis. It is not a general replacement for the [pandas DataFrame][pandas.DataFrame] or the [Labeled Data Storage][LabeledDataStorage] but a wrapper for the first one and an extension for the second one. Both and more data frames are accessible (or returned) by built-in methods. Therefore, using HEPDataStorage has no disadvantage.

### Assumptions ###
The basic ideas and assumption for the creations are:

* We should not care to much about the data type as the conversion is basically straight forward
* Data is read much more often than changed. And if it is changed, it should be saved to a [root tree](rootTree)
* We have a lot of other data that belongs to our actual root data
    * meta-data like name of the data, name of the columns used, plot-color...
    * additional data like labels, weights...
* Conversion of data takes neglectable cpu-time compared to the time used by the training, whereas the memory used to store several copies ( a [numpy array][numpy.array] and a [pandas Data Frame][pandas.DataFrame] ) of actually the same data can be significant.

* All parameters, data etc you set should be contained in one file, the config file.

## How to work with this package ##

### main ###
This module is exclusively to start your code. If you want to run your code multiple times (the *whole* code), this is the right place.

### physical_analysis ###
In this module, the actual analysis is going on. The methods should be just as long as they have to be to understand what is going on, but should consist of method calls, mainly to the ml_analysis module, without dirty work.

*every line should contain a _physical_ sentence*

for example: reweight the monte-carlo data and the real-data with a bins reweighter



[pandas.DataFrame]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[LabeledDataStorage]: http://yandex.github.io/rep/data.html#module-rep.data.storage
[numpy.array]: http://docs.scipy.org/doc/numpy-1.10.1/user/basics.rec.html
[rootTree]: https://root.cern.ch/doc/v606/classTTree.html