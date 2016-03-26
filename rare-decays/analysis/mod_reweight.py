# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:31:19 2016

@author: mayou
"""
import config as cnf
from tools.dev_tool import make_logger

import ROOT
from hep_ml import reweight
from root_numpy import root2array, rec2array, tree2rec

logger = make_logger(__name__)

def mc_real_reweight(mcdata, realdata):

    import config as cnf
    from tools.dev_tool import make_logger

    import ROOT
    from hep_ml import reweight
    from root_numpy import root2array, rec2array, tree2rec
    import matplotlib.pyplot as plt





    gbreweighter = reweight.BinsReweighter()
    gbreweighter.fit(mcdata, realdata)
    gb_weights = gbreweighter.predict_weights(mcdata)


    import numpy
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import roc_auc_score

    data = numpy.concatenate([mcdata, realdata])
    labels = numpy.array([0] * len(mcdata) + [1] * len(realdata))

    weights = {}
    weights['original'] = numpy.ones(len(mcdata))
    weights['bins'] = numpy.ones(len(realdata))
    weights['gb_weights'] = gb_weights
    print gb_weights

    list1 = []
    listreal = []
    listmc = []
    # plt.hist(mcdata[0:10000][0])
    #assert len(mcdata) == len(realdata), "error"
    for i in range(len(mcdata)):
        listmc.append(mcdata[i][0])

        list1.append(mcdata[i][0]*gb_weights[i])
    for i in range(len(realdata)):
        listreal.append(realdata[i][0])


    nbins = 200
    plt.subplot(1,3,1)
    plt.hist(listmc,nbins)
    plt.title("mcdata")
    plt.subplot(1,3,2)
    plt.hist(listreal,nbins)
    plt.title("realdata")
    plt.subplot(1,3,3)
    plt.hist(list1,nbins)
    plt.title("reweighted_data")
    plt.show()

#    for name, new_weights in weights.items():
#        W = numpy.concatenate([new_weights / new_weights.sum() * len(realdata), [1] * len(realdata)])
#        Xtr, Xts, Ytr, Yts, Wtr, Wts = train_test_split(data, labels, W, random_state=42, train_size=0.51)
#        clf = GradientBoostingClassifier(subsample=0.3, n_estimators=30).fit(Xtr, Ytr, sample_weight=Wtr)
#
#        print name, roc_auc_score(Yts, clf.predict_proba(Xts)[:, 1], sample_weight=Wts)









if __name__ == "__main__":
    import root_numpy
    import config as cnf
    import pandas
    import numpy
    import matplotlib.pyplot as plt
    data_mc = root_numpy.root2array(cnf.path_mc_reweight,"Bd2K1LL/DecayTree",
                                    cnf.branch_names)
    data_real = root_numpy.root2array(cnf.path_real_reweight,"Bd2K1LL/DecayTree",
                                    cnf.branch_names)
    #logger.debug(data_mc)

    original = pandas.DataFrame(data_mc)
    target = pandas.DataFrame(data_mc)
    #list_mc = [data_mc[i] for i in range(len(data_mc))]

    mc_weights = numpy.ones(len(data_mc))

    logger.debug(data_mc)
    nbins = 200
    plt.subplot(1,3,1)
    plt.hist(data_mc,nbins)
    plt.title("mcdata")
    plt.subplot(1,3,2)
    plt.hist(listreal,nbins)
    plt.title("realdata")
    plt.subplot(1,3,3)
    plt.hist(list1,nbins)
    plt.title("reweighted_data")
    plt.show()







#    fileRef = ROOT.TFile(cnf.path_mc_reweight)
#    tree = fileRef.Get("Bd2K1LL/DecayTree")
#    data_mc = tree2rec(tree, cnf.branch_names)
#
#    fileRef = ROOT.TFile(cnf.path_real_reweight)
#    tree = fileRef.Get("Bd2K1LL/DecayTree")
#    data_real = tree2rec(tree, cnf.branch_names)
#
#    reweight(data_mc, data_real)

"""  mc_data_reweight = root2array(cnf.path_mc_reweight, cnf.tree_real_reweight,
                                  cnf.branch_names)
    mc_data_reweight = rec2array(mc_data_reweight)
    real_data_reweight = root2array(cnf.path_real_reweight,
                                    cnf.tree_mc_reweight, cnf.branch_names)
    real_data_reweight = rec2array(real_data_reweight)
"""
#    dirSample = "/disk/data3/lhcb/rsilvaco/RareDecays/Bd2KpiEE/Ntuples/MonteCarlo/"+str(year)+"/Stripping-MCTruth/Bd2Kstee/"
#    fileRef = ROOT.TFile(dirSample+"/Bd2Kstee-MC-"+year+"-MagAll-B2KpiX2EEDarkBosonLine-MCTruth.root")
#    tree = fileRef.Get('DecayTree’)
#
#    arrayTree = tree2rec(tree, branches=['1 - sqrt(1 - '+particle+'_ProbNNe)', 'log('+particle+'_PT)', particle+'_TRACK_Eta',     'log(nTracks)'], selection='Polarity == '+polVal)
#    arrayTree.dtype.names = particle+'_ProbNNeLegPol', particle+'_PTLegPol', particle+'_EtaLegPol', 'nTracksLegPol'
logger.debug("finished!")