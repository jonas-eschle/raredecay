|Code Health| |Build Status| |PyPI version| |Dependency Status|

raredecay
=========

This package consists of several tools for the event selection of
particle decays, mostly built on machine learning techniques. It
contains:

-  a **data-container** holding data, weights, labels and more and
   implemented root-to-python data conversion as well as plots and
   KFold-data splitting
-  **reweighting** tools from the hep\_ml-repository wrapped in a
   KFolding structure and with metrics to evaluate the reweighting
   quality
-  **classifier optimization** tools for hyper-parameters as well as
   feature selection involving a backward-elimination
-  an **output handler** which makes it easy to add text as well as
   figures into your code and automatically save them to a file
-  ... and more

HowTo examples
--------------

To get an idea of the package, have a look at the howto notebooks: `HTML
version <https://mayou36.bitbucket.io/raredecay/howto/>`__ or the
`IPython
Notebooks <https://github.com/mayou36/raredecay/tree/master/howto>`__

Minimal example
---------------

Want to test whether your reweighting did overfit? Use train\_similar:

.. code:: python

    from raredecay.data import HEPDataStorage
    from raredecay.score import train_similar

    mc_data = HEPDataStorage(df, weights=*pd.Series weights*, target=0)
    real_data = HEPDataStorage(df, weights=*pd.Series weights*, target=1)

    score = train_similar(mc_data, real_data, old_mc_weights=1 *or whatever weights the mc had before*)

Getting started right now
-------------------------

If you want it the easy, fast way, have a look at the `Ready-to-use
scripts <https://github.com/mayou36/raredecay/tree/master/scripts_readyToUse>`__.
All you need to do is to have a look at every "TODO" task and probably
change them. Then you can run the script without the need of coding at
all.

Documentation and API
---------------------

The API as well as the documentation:
`Documentation <https://mayou36.bitbucket.io/raredecay/docs/>`__

Setup and installation
----------------------

Depending on which functionality you want to use, you may consider different installations.



Everything (reweighting incl. scoring, machine learning, ROOT bindings)
#######################################################################
Follow the instructions for each dependency separately and then install
raredecay via

::

    pip install raredecay[all]

Machine learning, advanced scores
#################################
First install the following version of REP (the -U can be omitted, but
is recommended to have the newest dependencies, on the other hand may
crashes REPs reproducibility):

::

    pip install -U https://github.com/yandex/rep/archive/stratifiedkfold.zip

Then install the package via

::

    pip install -U raredecay[ml]

Reweighting (without scoring)
#############################

To install the newest version of hep\_ml containing the
loss-regularization (recommended, but optional).

::

    pip install -U git+https://github.com/arogozhnikov/hep_ml.git

Then, install the raredecay package (without ROOT-support) via

::

    pip install raredecay

To make sure you can convert ROOT-NTuples, use

::

    pip install raredecay[root]  # *use raredecay\[root\] in a zsh-console*

As it is a young package still under developement, it may receive
regular updates and improvements and it is probably a good idea to
regularly download the newest package.

.. |Code Health| image:: https://landscape.io/github/mayou36/raredecay/master/landscape.svg?style=flat
   :target: https://landscape.io/github/mayou36/raredecay/master
.. |Build Status| image:: https://travis-ci.org/mayou36/raredecay.svg?branch=master
   :target: https://travis-ci.org/mayou36/raredecay
.. |PyPI version| image:: https://badge.fury.io/py/raredecay.svg
   :target: https://badge.fury.io/py/raredecay
.. |Dependency Status| image:: https://www.versioneye.com/user/projects/58273f1df09d22004f5914f9/badge.svg?style=flat-square
   :target: https://www.versioneye.com/user/projects/58273f1df09d22004f5914f9
