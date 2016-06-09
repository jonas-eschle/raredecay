# README of the raredecay-package #

# Overview #
 The repository is created for a particle decay analysis. The idea was to create a workplace where you can focus on the data manipulation and analysis instead of things like difficult data-conversion, plotting, making good output and more. The core consists of 
- a solid data-structure, a wrapper around pandas DataFrame which provides plotting function, data-naming, creation of sub-folds, weight-, label-storage and more.
- a machine learning analysis tool, which provides the most common used things for analysis like hyper-parameter optimization, classification etc. which works with the data-storage
- an output-module, which takes care of writing log-files, creating folders, saving images and more.
- the physical analysis module, where you can build your own analysis tools pretty easily.
- configurations files which contain all the necessary things to use the implemented physical analysis modes

The following is split into a (simple) user-manual, a more detailed user-with-developer-skills (create your own analysis tools) section and a developer section. Start reading with the simple user-manual and go on until you have enough details.

# A simple user-guide #

First of all, whatever you do: DO NEVER CHANGE THE KEYWORDS/VARIABLE NAMES, only its values.
Explained modules:
- main
- config-files
- meta_config

## main ##

The main file is the one to run. Here, you can specify what do you want to run by either directly invoking the *run* method with the right keyword (strings) or by comment and uncomment the right lines in the "if main"-body.

Next to the run-mode, the config-file should be specified. If None is given, the default one will be taken.

## config ##

The config file specifies everything about a certain run: which data to take, what classifier to use, how many iterations etc. This is where you basically control the run.

There are several config files under "run_config". Each of this can be used as well as your own can be created. Depending on the run-mode, several keywords have to be contained in the config-file in order to run correctly. Do understand the parameters, you best look at one of the existing files which contain a lot of explanation as comments.

### Administrative part ###

Every config file should contain some administrative part, which means the run_name, an additional message, an output configuration etc.

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