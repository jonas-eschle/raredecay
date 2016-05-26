# -*- coding: utf-8 -*-
"""
Created on Sat May 21 12:02:58 2016

@author: mayou
"""
from memory_profiler import profile

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter, OrderedDict

from rep.estimators.interface import Classifier

import hep_ml.reweight
from rep.metaml import ClassifiersFactory
from rep.utils import train_test_split
from rep.data import LabeledDataStorage

# classifier imports
from rep.estimators import SklearnClassifier, XGBoostClassifier, TMVAClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, BaggingClassifier
from rep.estimators.theanets import TheanetsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from rep.report import ClassificationReport

from raredecay import globals_
from raredecay.tools import dev_tool
from raredecay import meta_config

if __name__ == '__main__':
    logger = None  # dev_tool.make_logger(__name__)
else:
    # import configuration
    import importlib
    from raredecay import meta_config
    cfg = importlib.import_module(meta_config.run_config)
    logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


class Mayou(Classifier):
    """Classifier for raredecay analysis"""

    __DEFAULT_CLF_CFG = dict(
        xgb=dict(
            n_estimators=450,
            eta=0.1,
            subsample=0.9,
            bagging=None
        ),
        rdf=dict(
            n_estimators=1500,  # 1600
            max_features= 'auto', # only 1 feature seems to be pretty good...
            max_depth=120,
            min_samples_split=250,
            min_samples_leaf=150,
            min_weight_fraction_leaf=0.,
            max_leaf_nodes=None,
            bootstrap=False,
            oob_score=False,
            class_weight=None,
            bagging=None
        ),
#        erf=dict(
#            n_estimators=50,
#        ),
        nn=dict(
            layers=[200, 50],
            hidden_activation='logistic',
            output_activation='linear',
            input_noise=0.02,  # [0,1,2,3,4,5,10,20],
            hidden_noise=0,
            input_dropout=0,
            hidden_dropout=0,
            decode_from=1,
            weight_l1=0.01,
            weight_l2=0.03,
            scaler='standard',
            trainers=[{'optimize': 'nag', 'learning_rate': 0.1, 'min_improvement': 0.1}],
            bagging=None
        ),
#        ada=dict(
#            n_estimators=300,
#            learning_rate=0.1
#        ),
        gb=dict(
            learning_rate=0.05,
            n_estimators=500,
            max_depth=4,
            min_samples_split=600,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            subsample=0.8,
            max_features=None,
            max_leaf_nodes=None,
            bagging=None
        ),
    )
    __DEFAULT_BAG_CFG = dict(
        n_estimators=20,
        max_samples=0.9,
        max_features=1.0,
    )

    def __init__(self, base_estimators=None, bagging_base=10, stacking='xgb',
                 features_stack=None, bagging_stack=None, hunting=False):
        """blablabla


        Parameters
        ----------
        base_estimators : dict('clf': classifier OR keyword-parameters)
            Contains all the level-0 classifiers. The key is the name of the
            classifier and the value is either a **prefitted** classifier or
            a dictionary containing the keyword arguments to instantiate
            such a classifier.

            If no pre-trained classifier is provided, the key-value has to
            be one of the following:
             - **'xgb'** creates an XGBoost classifier
             - **'rdf'** creates a Random Forest classifier
             - **'erf'** creates a Forest consisting of Extra Randomized Trees
             - **'nn'** creates an artificial Neural Network from TheaNets
             - **'ada'** creates an AdaBoost instance with Decision Trees as
               basis
             - **'gb'** creates a Gradient Boosted classifier with Decision
               Trees as basis
        """
        self._base_estimators = OrderedDict(self.__DEFAULT_CLF_CFG) if base_estimators is None else base_estimators
        if isinstance(stacking, str):
            self._clf_1 = {stacking: None}
        elif isinstance(stacking, dict):
            self._clf_1 = stacking
        elif stacking in (False, None):
            stacking = False
        else:
            self._clf_1 = {'clf_stacking': stacking}  # stacking is a classifier
        self._bagging = bagging_base
        self._hunting = hunting
        self._clf_1_bagging = bagging_stack
        self._features_stack = features_stack
        self._clf_0 = {}
        self._factory = ClassifiersFactory()

    def get_params(self, deep=True):
        out = dict(
            base_estimators=None, bagging_base=10, stacking='xgb',
            features_stack=None, bagging_stack=None, hunting=False
            )
        return out

    def _transform(self, X):
        scaler = StandardScaler(copy=False)
        X = scaler.fit_transform(X)
        return X

    def _transform_pred(self, X):
        # TODO: change dummy method
        return self._transform(X)

    def _make_clf(self, clf, bagging=None):
        """Creates a classifier from a dict or returns the clf"""
        if isinstance(clf, dict):
            key, val = clf.popitem()
            try:
                val = self.__DEFAULT_CLF_CFG.get(key) if val is None else val
            except KeyError:
                logger.error(str(val) + " not an implemented classifier.")
                raise

            bagging = val.pop('bagging', bagging)


            if key == 'rdf':
                config_clf = dict(val)  # possible multi-threading arguments
                clf = SklearnClassifier(RandomForestClassifier(**config_clf))
            elif key == 'erf':
                config_clf = dict(val)  # possible multi-threading arguments
                clf = SklearnClassifier(ExtraTreesClassifier(**config_clf))
            elif key == 'nn':
                config_clf = dict(val)  # possible multi-threading arguments
                clf = TheanetsClassifier(**config_clf)
            elif key == 'ada':
                config_clf = dict(val)  # possible multi-threading arguments
                clf = SklearnClassifier(AdaBoostClassifier(**config_clf))
            elif key == 'gb':
                config_clf = dict(val)  # possible multi-threading arguments
                clf = SklearnClassifier(GradientBoostingClassifier(**config_clf))
            elif key == 'xgb':
                config_clf = dict(val)  # possible multi-threading arguments
                clf = XGBoostClassifier(**config_clf)
            elif hasattr(clf, 'fit'):
                bagging = False  # return the classifier

            # bagging over the instantiated estimators
            if isinstance(bagging, int) and bagging >= 1:
                bagging = dict(self.__DEFAULT_BAG_CFG, n_estimators=bagging)
            if isinstance(bagging, dict):
            # TODO: implement multi-thread:
                bagging.update({'base_estimator': clf})
                clf = SklearnClassifier(BaggingClassifier(**bagging))



        else:
            raise ValueError(str(clf) + " not valid as a classifier.")


        clf = {key: clf}
        return clf

    @profile
    def _factory_fit(self, X, y, sample_weight):

        X = self._transform(X)

        # create classifiers from initial dictionary
        if self._base_estimators != {}:
            for key, val in self._base_estimators.items():
                clf = self._make_clf({key: val}, bagging=self._bagging)
                self._clf_0.update(clf)
            self._base_estimators = {}

        # add base estimators to factory
        for key, val in self._clf_0.iteritems():
            self._factory.add_classifier(key, val)

        # parallel on factory level -> good mixture of clfs (one uses lot of RAM, one cpu...)
        parallel_profile = 'threads-' + str(min([len(self._factory.items()), globals_.free_cpus()]))

        # fit all classifiers
        print "start fitting factory"
        self._factory.fit(X, y, sample_weight, parallel_profile=parallel_profile)

        return self

    def _factory_predict(self, X):

        X = self._transform(X)

        # parallel on factory level -> good mixture of clfs (one uses lot of RAM, one cpu...)
        parallel_profile = 'threads-' + str(min([len(self._factory.items()), globals_.free_cpus()]))

        # predict, return a dictionary
        predictions = self._factory.predict(X, parallel_profile=parallel_profile)

        # slice the arrays of predictions in the dict right
        for key, val in predictions.items():
            predictions[key] = val[:,1]
        return pd.DataFrame(predictions)

    def _factory_predict_proba(self, X):

        X = self._transform(X)

        # parallel on factory level -> good mixture of clfs (one uses lot of RAM, one cpu...)
        parallel_profile = 'threads-' + str(min([len(self._factory.items()), globals_.free_cpus()]))

        # predict, return a dictionary
        predictions = self._factory.predict_proba(X, parallel_profile=parallel_profile)

        # slice the arrays of predictions in the dict right
        for key, val in predictions.items():
            predictions[key] = val[:,1]
        predictions = self._transform_pred(pd.DataFrame(predictions))
        return predictions

    def _get_X_stack(self, X):

        X = self._transform(X)

        # get the predictions of the base estimators
        lvl_0_proba = pd.DataFrame(self._factory_predict_proba(X))

        # add data features to stacking data
        if self._features_stack is not None:
            if self._features_stack == 'all':
                self._features_stack = self.features
            X_data = pd.DataFrame(X, columns=self._features_stack)
            lvl_0_proba = pd.concat([lvl_0_proba, X_data], axis=1, copy=False)

        return lvl_0_proba

    def _clf_1_fit(self, X_stack, y, sample_weight):


        if self._clf_1 not in (False, None):
            if self._clf_1.values()[0] is None or isinstance(self._clf_1.values()[0], dict):
                self._clf_1 = self._make_clf(self._clf_1, bagging=self._clf_1_bagging)

            self._clf = copy.deepcopy(self._clf_1.values()[0])

        self._clf.fit(X_stack, y, sample_weight)
    @profile
    def fit(self, X, y, sample_weight=None):

        self.features = X.columns.values

        self._factory_fit(X, y, sample_weight)

        X_stack = self._get_X_stack(X)

        self._clf_1_fit(X_stack, y, sample_weight)

        return self


#        if self._clf_1 is not None:
#            lvl_0_proba = factory.predict_proba(X, parallel_profile=parallel_profile)
#            for key, val in lvl_0_proba.items():
#                lvl_0_proba[key] = val[:,1]
#            lvl_0_proba = pd.DataFrame(lvl_0_proba)

    def predict(self, X):

        # TODO: inside get_X_stack: lvl_0_proba = self._factory_predict_proba(X)
        X_stack = self._get_X_stack(X)
        return self._clf.predict(X_stack)

    def predict_proba(self, X):
        X_stack = self._get_X_stack(X)
        return self._clf.predict_proba(X_stack)

    def test_on(self, X, y, sample_weight=None):
        lds = LabeledDataStorage(X, y, sample_weight)
        return self.test_on_lds(lds)

    def test_on_lds(self, lds):
        return ClassificationReport({'Mayou clf': self}, lds)
    def staged_predict_proba(self, X):
        try:
            X_stack = self._get_X_stack(X)
            temp_proba = self._clf.staged_predict_proba(X_stack)
        except:
            print "error occured in mayou, staged predict proba not supported"
        return temp_proba

if __name__ == '__main__':
    """selftest"""
    import matplotlib.pyplot as plt

    from rep.metaml import FoldingClassifier
    from rep.report.metrics import RocAuc, significance, ams, OptimalAccuracy, OptimalAMS
    from sklearn.svm import NuSVC

    from root_numpy import root2array, rec2array

    folding = False
    higgs_data = True

    n_ones, n_zeros = 3000, 3000
    n_tot = n_ones + n_zeros
    feature_one = np.concatenate((np.random.normal(loc=0.2, size=n_ones), np.random.exponential(scale=1.0, size=n_zeros)))
    feature_two = np.concatenate((np.random.exponential(scale=1.7, size=n_ones), np.random.normal(loc=-0.7, size=n_zeros)))
    X = pd.DataFrame({'one': feature_one, 'two': feature_two})
    y = np.concatenate((np.ones(n_ones), np.zeros(n_zeros)))
    w = np.ones(n_tot)

    if higgs_data:

#        missing energy magnitude, missing energy phi,
#        jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag,
#        jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag,
#        jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag,
#        jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag,
#        m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb,
        branch_names = """lepton pT, lepton eta, lepton phi,
        m_wwbb""".split(",")
        branch_names = [c.strip() for c in branch_names]
        branch_names = (b.replace(" ", "_") for b in branch_names)
        branch_names = list(b.replace("-", "_") for b in branch_names)

        signal = root2array("/home/mayou/Downloads/higgs/HIGGSsignal.root",
                            "tree",
                            branch_names)
        signal = pd.DataFrame(rec2array(signal))

        backgr = root2array("/home/mayou/Downloads/higgs/HIGGSbackground.root",
                            "tree",
                            branch_names)
        backgr = pd.DataFrame(rec2array(backgr))

        # for sklearn data is usually organised
        # into one 2D array of shape (n_samples x n_features)
        # containing all the data and one array of categories
        # of length n_samples
        X = pd.concat((signal, backgr))
        y = np.concatenate((np.ones(signal.shape[0]),
                            np.zeros(backgr.shape[0])))
        w = np.ones(len(X))



    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.33)


    lds = LabeledDataStorage(X_test, y_test, w_test)
    #clf = SklearnClassifier(RandomForestClassifier())
    #clf_stacking = XGBoostClassifier(n_estimators=700, eta=0.1, nthreads=8)
    clf_stacking='nn'
    clf = Mayou(bagging_base=None, bagging_stack=None, stacking=clf_stacking)#, features_stack=branch_names)
    #clf = XGBoostClassifier(n_estimators=350, eta=0.1, nthreads=8)
    #clf = SklearnClassifier(BaggingClassifier(clf, max_samples=0.8))
    #clf = SklearnClassifier(NuSVC(cache_size=1000000))
    if folding:
        X_train = X_test = X
        y_train = y_test = y
        w_train = w_test = w
        clf = FoldingClassifier(clf, n_folds=5)


    clf.fit(X_train, y_train, w_train)
    print "Predictions: ", clf.predict(X_test)
    print "Probabilites", clf.predict_proba(X_test)
    #report = clf.test_on(X_test, y_test, w_test)
    report = clf.test_on_lds(lds)
    report.roc().plot(new_plot=True)
    print "\nROC AUC = ", report.compute_metric(RocAuc())
    print "\nOptimalAccuracy = ", report.compute_metric(OptimalAccuracy())
    print "\nOptimalAMS = ", report.compute_metric(OptimalAMS())
    print "score: ", clf.score(X_test, y_test, w_test)
#    report.feature_importance_shuffling().plot(new_plot=True)
#    report.features_correlation_matrix().plot(new_plot=True)

    plt.show()