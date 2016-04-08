# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

# What is it for?
 The repository is created for a particle decay analysis. The core is a solid data-structure, a wrapper around pandas DataFrame and more, with wrappers for different machine-learning tools including binary classifiers, reweighters and more.

## The core: _hepDataFrame_
 The idea is to create a data frame which perfectly fits the need for data analysis. It is not a general replacement for the [pandas DataFrame][pandas.DataFrame] or the [Labeled Data Storage][LabeledDataStorage] but a wrapper for the first one and an extension for the second one.
### Assumptions 
The basic ideas and assumption for the creations are:
* We should not care to much about the data type as it is basically straight forward
* Data is 


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