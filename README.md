# raredecay #

This package consists of several tools for the event selection of particle decays, mostly built on machine learning techniques.
It contains:

- a **data-container** holding data, weights, labels and more and implemented root-to-python data conversion as well as plots and KFold-data splitting
- **reweighting** tools from the hep_ml-repository wrapped in a KFolding structure and with metrics to evaluate the reweighting quality
- **classifier optimization** tools for hyper-parameters as well as feature selection involving a backward-elimination
- an **output handler** which makes it easy to add text as well as figures into your code and automatically save them to a file
- ... and more

## HowTo examples ##

To get an idea of the package, have a look at the howto notebooks:
[HTML version](http://mayou36.bitbucket.org/raredecay/howto/) or the 
[IPython Notebooks](https://github.com/mayou36/raredecay/tree/master/howto)

## Documentation and API ##

The API as well as the documentation:
[Documentation](http://mayou36.bitbucket.org/raredecay/docs/)

## Setup and installation ##

PIP package will may be uploaded in the future. Until then, use the following.
first install the very newest version of REP  
(the -U can be omitted if necessary, but is recommended):
```
pip install -U https://github.com/yandex/rep/archive/stratifiedkfold.zip
```
Then, install the raredecay package via
```
pip install git+https://github.com/mayou36/raredecay.git
```
As it is a young package still under developement, it may receive regular updates and improvements and it is probably a good idea to regularly download the newest package.


[pandas.DataFrame]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[LabeledDataStorage]: http://yandex.github.io/rep/data.html#module-rep.data.storage
[numpy.array]: http://docs.scipy.org/doc/numpy-1.10.1/user/basics.rec.html
[rootTree]: https://root.cern.ch/doc/v606/classTTree.html
