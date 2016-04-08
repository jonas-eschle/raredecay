# README #

This README would normally document whatever steps are necessary to get your application up and running.

# What is it for? #
 The repository is created for a particle decay analysis. The core is a solid data-structure, a wrapper around pandas DataFrame and more, with wrappers for different machine-learning tools including binary classifiers, reweighters and more.

## The core: _HEPDataStorage_ ##
 The idea is to create a data frame which perfectly fits the need for data analysis. It is not a general replacement for the [pandas DataFrame][pandas.DataFrame] or the [Labeled Data Storage][LabeledDataStorage] but a wrapper for the first one and an extension for the second one. Both and more data frames are accessible (or returned) by built-in methods. Therefore, using HEPDataStorage has no disadvantage.
### Assumptions ###
The basic ideas and assumption for the creations are:

* We should not care to much about the data type as the conversion is basically straight forward
* Data is read much more often than changed. And if it is changed, it should be saved to a [root tree](rootTree)
* We have a lot of other data that belongs to our actual root data
 * meta-data like name of the data, name of the columns used, plot-color...
 * additional data like labels, weights...
* Conversion of data takes neglectable cpu-time compared to the time used by the training, whereas the memory used to store several copies ( a [numpy array][numpy.array] and a [pandas Data Frame][pandas.DataFrame] ) of actually the same data is significant. At least on an average notebook.



### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact

[pandas.DataFrame]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[LabeledDataStorage]: http://yandex.github.io/rep/data.html#module-rep.data.storage
[numpy.array]: http://docs.scipy.org/doc/numpy-1.10.1/user/basics.rec.html
[rootTree]: https://root.cern.ch/doc/v606/classTTree.html