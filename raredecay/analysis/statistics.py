# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 20:00:33 2016

@author: Jonas Eschle 'Mayou36'

This modul contains several tools like fits.
"""
from __future__ import division, absolute_import
import numpy as np

import ROOT
from ROOT import RooRealVar, RooArgList, RooArgSet, RooAddPdf, RooDataSet, RooAbsReal
from ROOT import RooFit, RooCBShape, RooExponential
from ROOT import RooGaussian, RooMinuit
from ROOT import TCanvas  # HACK to prevent not plotting canvas by root_numpy import. BUG.
from root_numpy import array2tree
from ROOT import RooCategory, RooUnblindPrecision

from raredecay.globals_ import out

# from raredecay import meta_config


def fit_mass(data, column='B_M', n_bkg=None, n_sig=None, blind=False,
             second_storage=None, nll_profile=False, plot_verbosity=3):
    """Fit a pdf to a distribution"""

    if not (isinstance(column, str) or len(column) == 1):
        raise ValueError("Fitting to several columns " + str(column) + " not supported.")
    if blind is not False:
        lower_blind, upper_blind = blind
        blind = True

    # create data
    data_array, _t1, _t2 = data.make_dataset(second_storage, columns=column)
    del _t1, _t2

    # double crystalball variables
    min_x, max_x = min(data_array[column]), max(data_array[column])

    x = RooRealVar("x", "x variable", min_x, max_x)

    # create data
    data_array = np.array([i[0] for i in data_array.as_matrix()])
    data_array.dtype = [('x', np.float64)]
    tree1 = array2tree(data_array, "x")
    data = RooDataSet("data", "Data", RooArgSet(x), RooFit.Import(tree1))

    # TODO: export somewhere? does not need to be defined inside...
    mean = RooRealVar("mean", "Mean of Double CB PDF", 5280, 5100, 5600)#, 5300, 5500)
    sigma = RooRealVar("sigma", "Sigma of Double CB PDF", 40, 0.001, 200)
    alpha_0 = RooRealVar("alpha_0", "alpha_0 of one side", 5.715)#, 0, 150)
    alpha_1 = RooRealVar("alpha_1", "alpha_1 of other side", -4.019)#, -200, 0.)
    lambda_0 = RooRealVar("lambda_0", "Exponent of one side", 3.42)#, 0, 150)
    lambda_1 = RooRealVar("lambda_1", "Exponent of other side", 3.7914)#, 0, 500)

    # TODO: export somewhere? pdf construction
    frac = RooRealVar("frac", "Fraction of crystal ball pdfs", 0.479)

    crystalball1 = RooCBShape("crystallball1", "First CrystalBall PDF", x,
                              mean, sigma, alpha_0, lambda_0)
    crystalball2 = RooCBShape("crystallball2", "Second CrystalBall PDF", x,
                              mean, sigma, alpha_1, lambda_1)
    doubleCB = RooAddPdf("doubleCB", "Double CrystalBall PDF",
                         crystalball1, crystalball2, frac)

#    n_sig = RooRealVar("n_sig", "Number of signals events", 10000, 0, 1000000)

    # test input
    if n_sig == n_bkg == 0:
        raise ValueError("n_sig as well as n_bkg is 0...")

    if n_bkg is None:
        n_bkg = RooRealVar("n_bkg", "Number of background events", 10000, 0, 500000)
    elif n_bkg >= 0:
        n_bkg = RooRealVar("n_bkg", "Number of background events", int(n_bkg))
    else:
        raise ValueError("n_bkg is not >= 0 or None")

    if n_sig is None:
        n_sig = RooRealVar("n_sig", "Number of signal events", 1050, 0, 20000)

        # START BLINDING
        blind_cat = RooCategory("blind_cat", "blind state category")
        blind_cat.defineType("unblind", 0)
        blind_cat.defineType("blind", 1)
        if blind:
            blind_cat.setLabel("blind")
            blind_n_sig = RooUnblindPrecision("blind_n_sig", "blind number of signals",
                                              "wasistdas", n_sig.getVal(), 10000, n_sig, blind_cat)
        else:
#            blind_cat.setLabel("unblind")
            blind_n_sig = n_sig

        print "n_sig value " + str(n_sig.getVal())
#        raw_input("blind value " + str(blind_n_sig.getVal()))

#        n_sig = blind_n_sig



        # END BLINDING
    elif n_sig >= 0:
        n_sig = RooRealVar("n_sig", "Number of signal events", int(n_sig))
    else:
        raise ValueError("n_sig is not >= 0")

#    if not blind:
#        blind_n_sig = n_sig

    # create bkg-pdf
    lambda_exp = RooRealVar("lambda_exp", "lambda exp pdf bkg", -0.1, -1000., 1000.)
    bkg_pdf = RooExponential("bkg_pdf", "Background PDF exp", x, lambda_exp)

    if blind:
        comb_pdf = RooAddPdf("comb_pdf", "Combined DoubleCB and bkg PDF",
                             RooArgList(doubleCB, bkg_pdf), RooArgList(blind_n_sig, n_bkg))
    else:
        comb_pdf = RooAddPdf("comb_pdf", "Combined DoubleCB and bkg PDF",
                             RooArgList(doubleCB, bkg_pdf), RooArgList(n_sig, n_bkg))

    # create test dataset
#    mean_gauss = RooRealVar("mean_gauss", "Mean of Gaussian", 5553, -10000, 10000)
#    sigma_gauss = RooRealVar("sigma_gauss", "Width of Gaussian", 20, 0.0001, 300)
#    gauss1 = RooGaussian("gauss1", "Gaussian test dist", x, mean_gauss, sigma_gauss)
#    lambda_data = RooRealVar("lambda_data", "lambda exp data", -.002)
#    exp_data = RooExponential("exp_data", "data example exp", x, lambda_data)
#    frac_data = RooRealVar("frac_data", "Fraction PDF of data", 0.15)
#
#    data_pdf = RooAddPdf("data_pdf", "Data PDF", gauss1, exp_data, frac_data)
#    data = data_pdf.generate(RooArgSet(x), 30000)


#    data.printValue()
#    xframe = x.frame()
#    data_pdf.plotOn(xframe)

#    comb_pdf.fitTo(data, RooFit.Extended(ROOT.kTRUE), RooFit.NumCPU(meta_config.get_n_cpu()))
    # HACK to get 8 cores in testing
    result_fit = comb_pdf.fitTo(data, RooFit.Extended(ROOT.kTRUE), RooFit.NumCPU(12))
    # HACK end

    if plot_verbosity >= 3:
        c2 = TCanvas("c2", "first canvas")
        c2.cd()
        x_frame = x.frame()
    if blind:
        x.setRange("lower", min_x, lower_blind)
        x.setRange("upper", upper_blind, max_x)
        range_str = "lower,upper"
        lower_cut_str = str(min_x) + "<=" + column + "&&" + column + "<=" + str(lower_blind)
        upper_cut_str = str(upper_blind) + "<=" + column + "&&" + column + "<=" + str(max_x)
        sideband_cut_str = "("+lower_cut_str+")" + "||" + "("+upper_cut_str+")"

        n_entries = data.reduce(sideband_cut_str).numEntries() / data.numEntries()
#        raw_input("n_entries: " + str(n_entries))
        if plot_verbosity >= 3:
            data.plotOn(x_frame, RooFit.CutRange(range_str), RooFit.NormRange(range_str))
            comb_pdf.plotOn(x_frame, RooFit.Range(range_str),
                            RooFit.Normalization(n_entries, RooAbsReal.Relative),
                            RooFit.NormRange(range_str))
    else:
        if plot_verbosity >= 3:
            data.plotOn(x_frame)
            comb_pdf.plotOn(x_frame)


    if plot_verbosity >= 3:
        x_frame.Draw()

    print "n_sig value and type: ", n_sig.getVal(), type(n_sig)
    print "blind_n_sig value and type: ", n_sig.getVal(), type(blind_n_sig)

#    raw_input("")

    if not blind and nll_profile:

#        nll_range = RooRealVar("nll_range", "Signal for nLL", n_sig.getVal(),
#                               -10, 2 * n_sig.getVal())
        sframe = n_sig.frame(RooFit.Bins(20), RooFit.Range(1, 1000))
        # HACK for best n_cpu
        lnL = comb_pdf.createNLL(data, RooFit.NumCPU(8))
        # HACK end
        lnProfileL = lnL.createProfile(ROOT.RooArgSet(n_sig))
        lnProfileL.plotOn(sframe, RooFit.ShiftToZero())
        c4 = TCanvas("c4", "NLL Profile")
        c4.cd()
        sframe.Draw()

    if blind:
        return -9999
    else:
        return n_sig.getVal()


#    nll_plot = RooRealVar("nll_plot", "NLL plotting range", 0.01, 0.99)
#    nll_frame = nll_plot.frame()
#    my_nll = comb_pdf.createNLL(data, RooFit.NumCPU(8))
#    RooMinuit(my_nll).migrad()
#    my_nll.plotOn(nll_frame)
#    nll_frame.Draw()
#    data.plotOn(xframe)
#    comb_pdf.plotOn(xframe)
#    xframe.Draw()

#    return xframe

    params = comb_pdf.getVariables()
    params.Print("v")

#    if blind:
#        print("Value of n_sig:", blind_n_sig.evaluate())


if __name__ == '__main__':
    data = "empty"
#    data = RooDataSet("data", )
    from raredecay.tools.data_storage import HEPDataStorage
    import pandas as pd
    import matplotlib.pyplot as plt
#    np.random.seed(40)

    n_sig_fit = []
    n_sig_gen = range(0, 500, 5)
    for n_sig in n_sig_gen:
#        n_sig = 1000
        data = pd.DataFrame(np.random.normal(loc=5280, scale=40, size=(n_sig, 2)), columns=['x', 'y'])
        bkg_data = np.array([i for i in (np.random.exponential(scale=4800,
                                         size=(30000, 2)) + 5000) if i[0] < 6000])
        data = pd.concat([data, pd.DataFrame(bkg_data, columns=['x', 'y'])], ignore_index=True)
        data = HEPDataStorage(data)
        n_sig_averaged = np.mean([fit_mass(data=data, column='x', blind=False, plot_verbosity=1) for i in range(10)])
        n_sig_fit.append(n_sig_averaged)#[5300, 5500]))
        print n_sig_fit
#        raw_input("")
    plt.plot(np.array(n_sig_gen), np.array(n_sig_fit), linestyle='--', marker='o')
    plt.show()

    print "finished"
