# -*- coding: utf-8 -*-
"""
Created on Sat May 21 12:02:58 2016

@author: Jonas Eschle "Mayou36"
"""


import copy
import numpy as np
import pandas as pd
# import seaborn as sns
from collections import OrderedDict

from rep.estimators.interface import Classifier

from rep.metaml import ClassifiersFactory
from rep.utils import train_test_split
from rep.data import LabeledDataStorage

# classifier imports
from rep.estimators import SklearnClassifier, XGBoostClassifier  # , TMVAClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier  # , VotingClassifier
from rep.estimators.theanets import TheanetsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifie
from sklearn.preprocessing import StandardScaler

from rep.report import ClassificationReport

from raredecay import globals_
from raredecay.tools import dev_tool
from raredecay import meta_config

import importlib
cfg = importlib.import_module(meta_config.run_config)
logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)


# TODO: Transformations don't work
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
            n_estimators=1600,  # 1600
            max_features='auto',  # only 1 feature seems to be pretty good...
            max_depth=200,
            min_samples_split=250,
            min_samples_leaf=150,
            min_weight_fraction_leaf=0.,
            max_leaf_nodes=None,
            bootstrap=False,
            oob_score=False,
            n_jobs=7,
            class_weight=None,
            bagging=None
        ),

        nn=dict(
            layers=[100, 100],
            hidden_activation='logistic',
            output_activation='linear',
            input_noise=0,  # [0,1,2,3,4,5,10,20],
            hidden_noise=0,
            input_dropout=0,
            hidden_dropout=0.05,
            decode_from=1,
            weight_l1=0.01,
            weight_l2=0.03,
            scaler='standard',
            trainers=[{'optimize': 'adagrad', 'patience': 2, 'momentum': 0.5, 'nesterov': True,
                       'learning_rate': 0.2, 'min_improvement': 0.01}],
            bagging=None
        ),
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

    def __init__(self, base_estimators=None, bagging_base=None, stacking='xgb',
                 features_stack=None, bagging_stack=None, hunting=False,
                 transform=True, transform_pred=True):
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
        if base_estimators is None:
            OrderedDict(self.__DEFAULT_CLF_CFG)
        else:
            self._base_estimators = base_estimators
        if isinstance(stacking, str):
            self._clf_1 = {stacking: None}
        elif isinstance(stacking, dict):
            self._clf_1 = stacking
        elif stacking in (False, None):
            stacking = False
        else:
            self._clf_1 = {'clf_stacking': stacking}  # stacking is a classifier

        self._transform_data = transform
        self._bagging = bagging_base
        self._hunting = hunting
        self._clf_1_bagging = bagging_stack
        self._features_stack = features_stack
        self._clf_0 = {}
        self._factory = ClassifiersFactory()
        self._base_scaler = None
        self._pred_scaler = None

    def get_params(self, deep=True):
        out = dict(
            base_estimators=None, bagging_base=None, stacking='xgb',
            features_stack=None, bagging_stack=None, hunting=False
            )
        return out

    def _transform(self, X, fit=False):

        if self._transform_data:
            columns = copy.deepcopy(X.keys())
            index = copy.deepcopy(X.index)

            if fit:
                self._base_scaler = StandardScaler(copy=True)
                self._base_scaler.fit_transform(X)
            else:
                X = self._base_scaler.transform(X)

            X = pd.DataFrame(X, index=index, columns=columns)
        return X

    def _transform_pred(self, X, fit=False):

        if self._transform_pred:
            columns = copy.deepcopy(X.keys())
            index = copy.deepcopy(X.index)

            if fit:
                self._pred_scaler = StandardScaler(copy=True)  # don't change data!
                self._pred_scaler.fit_transform(X)
            else:
                X = self._pred_scaler.transform(X)

            X = pd.DataFrame(X, index=index, columns=columns)
        return X

    def _make_clf(self, clf, bagging=None):
        """Creates a classifier from a dict or returns the clf"""
        if isinstance(clf, dict):
            key, val = clf.popitem()
            try:
                val = self.__DEFAULT_CLF_CFG.get(key) if val is None else val
            except KeyError:
                logger.error(str(val) + " not an implemented classifier.")
                raise

            temp_bagging = val.pop('bagging', bagging)
            bagging = temp_bagging if bagging is None else bagging

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

    def _factory_fit(self, X, y, sample_weight):

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
        parallel_profile = 'threads-' + str(min([len(self._factory.items()),
                                                 globals_.free_cpus()]))

        # fit all classifiers
        print "start fitting factory"
        self._factory.fit(X, y, sample_weight, parallel_profile=parallel_profile)

        return self

    def _factory_predict(self, X):

        columns = copy.deepcopy(X.keys())
        index = copy.deepcopy(X.index)

        # parallel on factory level -> good mixture of clfs (one uses lot of RAM, one cpu...)
        parallel_profile = 'threads-' + str(min([len(self._factory.items()),
                                                 globals_.free_cpus()]))

        # predict, return a dictionary
        predictions = self._factory.predict(X, parallel_profile=parallel_profile)

        # slice the arrays of predictions in the dict right
        for key, val in predictions.items():
            predictions[key] = val[:, 1]
        return pd.DataFrame(predictions, index=index, columns=columns)

    # @profile
    def _factory_predict_proba(self, X):

        index = X.index

        # parallel on factory level -> good mixture of clfs (one uses lot of RAM, one cpu...)
        parallel_profile = 'threads-' + str(min([len(self._factory.items()),
                                                 globals_.free_cpus()]))
        print parallel_profile
        parallel_profile = None
        # predict, return a dictionary
        predictions = self._factory.predict_proba(X, parallel_profile=parallel_profile)

        # slice the arrays of predictions in the dict right
        for key, val in predictions.items():
            predictions[key] = val[:, 1]
        return pd.DataFrame(predictions, index=index)

    def _get_X_stack(self, X, fit_scaler=False):

        # get the predictions of the base estimators
        lvl_0_proba = pd.DataFrame(self._factory_predict_proba(X), index=X.index,
                                   columns=self._factory.keys())
        lvl_0_proba = self._transform_pred(lvl_0_proba, fit=fit_scaler)

        # add data features to stacking data
        if self._features_stack is not None:
            if self._features_stack == 'all':
                self._features_stack = self.features
            elif not set(self._features_stack).issubset(self.features):
                raise RuntimeError("Stacked features not in features of the data fitted to")

            X_data = pd.DataFrame(X, columns=self._features_stack)
            lvl_0_proba = pd.concat([lvl_0_proba, X_data], axis=1, copy=False)

        return lvl_0_proba

    def _clf_1_fit(self, X, y, sample_weight):

        X_stack = self._get_X_stack(X, fit_scaler=True)

        if self._clf_1 not in (False, None):
            if self._clf_1.values()[0] is None or isinstance(self._clf_1.values()[0], dict):
                self._clf_1 = self._make_clf(self._clf_1, bagging=self._clf_1_bagging)

            self._clf = copy.deepcopy(self._clf_1.values()[0])

        self._clf.fit(X_stack, y, sample_weight)

    def _set_features(self, X):
        """Set the 'features' attribute for the classifier"""
        if isinstance(X, pd.DataFrame):
            self.features = X.columns.values
        else:
            self.features = ["Feature_" + str(i) for i in range(X.shape[1])]

    def fit(self, X, y, sample_weight=None):

        # initiate properties
        self._set_features(X)
        self.classes_ = range(len(set(y)))
        X = self._transform(X, fit=True)

        # fit the base estimators
        self._factory_fit(X, y, sample_weight)

        # fit the stacking classifier
        self._clf_1_fit(X, y, sample_weight)

        return self

    def predict(self, X):

        # TODO: inside get_X_stack: lvl_0_proba = self._factory_predict_proba(X)
        X = self._transform(X)
        X_stack = self._get_X_stack(X)
        return self._clf.predict(X_stack)

    def predict_proba(self, X):
        X = self._transform(X)
        X_stack = self._get_X_stack(X)
        return self._clf.predict_proba(X_stack)

    def test_on(self, X, y, sample_weight=None):
        lds = LabeledDataStorage(X, y, sample_weight)
        return self.test_on_lds(lds)

    def test_on_lds(self, lds):
        lds.data = self._transform(lds.data)
        return ClassificationReport({'Mayou clf': self}, lds)

    def staged_predict_proba(self, X):
        X = self._transform(X)
        X_stack = self._get_X_stack(X)
        try:
            temp_proba = self._clf.staged_predict_proba(X_stack)
        # TODO: change error catching
        except:
            print "error occured in mayou, staged predict proba not supported"
        return temp_proba

    def stacker_test_on_lds(self, lds):
        """Return report for the stacker only"""
        lds.data = self._get_X_stack(self._transform(lds.data))
        return ClassificationReport({'Mayou stacker': self._clf}, lds)

    def stacker_test_on(self, X, y, sample_weight=None):
        """Return report for the stacker only"""
        lds = LabeledDataStorage(X, y, sample_weight)
        return self.stacker_test_on_lds(lds)


if __name__ == '__main__':
    # selftest
    import matplotlib.pyplot as plt

    from rep.metaml import FoldingClassifier
#    from rep.report.metrics import RocAuc, ams, OptimalAccuracy, OptimalAMS  # , significance
#    from raredecay.tools.metrics import punzi_fom, precision_measure
#    from sklearn.svm import NuSVC
#    from sklearn.naive_bayes import GaussianNB

    from root_numpy import root2array, rec2array

    folding = True
    higgs_data = False
    primitiv = True

    n_ones, n_zeros = 20000, 20000
    n_tot = n_ones + n_zeros

    branch_names = ['two', 'one']
    feature_one = np.concatenate((np.random.normal(loc=0.2, size=n_ones),
                                  np.random.exponential(scale=1.0, size=n_zeros)))
    feature_two = np.concatenate((np.random.exponential(scale=1.7, size=n_ones),
                                  np.random.normal(loc=-0.7, size=n_zeros)))
    # feature_two = copy.deepcopy(feature_one)
    X = pd.DataFrame({'one': feature_one, 'two': feature_two})
    y = np.concatenate((np.ones(n_ones), np.zeros(n_zeros)))
    w = np.ones(n_tot)

    if higgs_data:

        #                jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag,
        #        jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag,
        #        jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag,
        #        jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag,
        branch_names = """
        missing energy magnitude, missing energy phi,
        m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb,
        lepton pT, lepton eta, lepton phi,
        m_wwbb""".split(",")
        branch_names = [c.strip() for c in branch_names]
        branch_names = (b.replace(" ", "_") for b in branch_names)
        branch_names = list(b.replace("-", "_") for b in branch_names)

        signal = root2array("/home/mayou/Downloads/higgs/HIGGSsignal.root",
                            "tree",
                            branch_names)
        signal = pd.DataFrame(rec2array(signal), columns=branch_names)

        backgr = root2array("/home/mayou/Downloads/higgs/HIGGSbackground.root",
                            "tree",
                            branch_names)
        backgr = pd.DataFrame(rec2array(backgr), columns=branch_names)

        signal = signal[:20000]
        backgr = backgr[:20000]

        # for sklearn data is usually organised
        # into one 2D array of shape (n_samples x n_features)
        # containing all the data and one array of categories
        # of length n_samples
        X = pd.concat((signal, backgr))
        y = np.concatenate((np.ones(signal.shape[0]),
                            np.zeros(backgr.shape[0])))
        w = np.ones(len(X))

    if primitiv:
        X = pd.DataFrame({'odin': np.array([2., 2., 2., 2., 3., 3., 2., 3., 8.,
                                            7., 8., 7., 8., 8., 7., 8.]),
                          'dwa': np.array([2.2, 2.1, 2.2, 2.3, 3.1, 3.1, 2.1, 3.2, 8.1,
                                           7.5, 8.2, 7.1, 8.5, 8.2, 7.6, 8.1])
                          })
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        w = np.ones(16)
        branch_names = ['odin', 'dwa']
    print branch_names
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.33)

    lds = LabeledDataStorage(X_test, y_test, w_test)
    # CLASSIFIER
    clf_stacking = SklearnClassifier(RandomForestClassifier(n_estimators=5000, bootstrap=False,
                                                            n_jobs=7))
    # clf_stacking = XGBoostClassifier(n_estimators=700, eta=0.1, nthreads=8,
    #                                 subsample=0.5
    #                                 )
    # clf_stacking='nn'
    clf = Mayou(base_estimators={'xgb': None}, bagging_base=None, bagging_stack=8,
                stacking=clf_stacking, features_stack=branch_names,
                transform=False, transform_pred=False)
    # clf = SklearnClassifier(GaussianNB())
    # clf = SklearnClassifier(BaggingClassifier(n_jobs=1, max_features=1.,
    # bootstrap=False, base_estimator=clf, n_estimators=20, max_samples=0.1))
    # clf = XGBoostClassifier(n_estimators=400, eta=0.1, nthreads=6)
    # clf = SklearnClassifier(BaggingClassifier(clf, max_samples=0.8))
    # clf = SklearnClassifier(NuSVC(cache_size=1000000))
    # clf = SklearnClassifier(clf)
    if folding:
        X_train = X_test = X
        y_train = y_test = y
        w_train = w_test = w
        clf = FoldingClassifier(clf, n_folds=5)

    clf.fit(X_train, y_train, w_train)


#    report.features_correlation_matrix().plot(new_plot=True)

    plt.show()
