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

from raredecay import meta_config

import matplotlib.pyplot as plt
# Bug fixing below
#plt.figure("TMPONLYTOFIGHTASTRANGEBUG")
#plt.plot([1, 2, 3])
#plt.close("TMPONLYTOFIGHTASTRANGEBUG")
# Bug fix end

def fit_mass(data, column, x, sig_pdf=None, bkg_pdf=None, n_sig=None, n_bkg=None,
             blind=False, nll_profile=False, second_storage=None, log_plot=False,
             pulls=True, sPlot=False,
             bkg_in_region=False, importance=3, plot_importance=3):
    """Fit a given pdf to a variable distribution


    Parameter
    ---------
    data : |hepds_type|
        The data containing the variable to fit to
    column : str
        The name of the column to fit the pdf to
    sig_pdf : RooFit pdf
        The signal Probability Density Function. The variable to fit to has
        to be named 'x'.
    bkg_pdf : RooFit pdf
        The background Probability Density Function. The variable to fit to has
        to be named 'x'.
    n_sig : None or numeric
        The number of signals in the data. If it should be fitted, use None.
    n_bkg : None or numeric
        The number of background events in the data.
        If it should be fitted, use None.
    blind : boolean or tuple(numberic, numberic)
        If False, the data is fitted. If a tuple is provided, the values are
        used as the lower (the first value) and the upper (the second value)
        limit of a blinding region, which will be omitted in plots.
        Additionally, no true number of signal will be returned but only fake.
    nll_profile : boolean
        If True, a Negative Log-Likelihood Profile will be generated. Does not
        work with blind fits.
    second_storage : |hepds_type|
        A second data-storage that will be concatenated with the first one.
    importance : |importance_type|
        |importance_docstring|
    plot_importance : |plot_importance_type|
        |plot_importance_docstring|

    Return
    ------
    tuple(numerical, numerical)
        Return the number of signals and the number of backgrounds in the
        signal-region. If a blind fit is performed, the signal will be a fake
        number. If no number of background events is required, -999 will be
        returned.
    """

    if not (isinstance(column, str) or len(column) == 1):
        raise ValueError("Fitting to several columns " + str(column) + " not supported.")
    if type(sig_pdf) == type(bkg_pdf) == None:
        raise ValueError("sig_pdf and bkg_pdf are both None-> no fit possible")
    if blind is not False:
        lower_blind, upper_blind = blind
        blind = True

    n_bkg_below_sig = -999
    # create data
    data_name = data.name
    data_array, _t1, _t2 = data.make_dataset(second_storage, columns=column)
    del _t1, _t2

    # double crystalball variables
    min_x, max_x = min(data_array[column]), max(data_array[column])

#    x = RooRealVar("x", "x variable", min_x, max_x)

    # create data
    data_array = np.array([i[0] for i in data_array.as_matrix()])
    data_array.dtype = [('x', np.float64)]
    tree1 = array2tree(data_array, "x")
    data = RooDataSet("data", "Data", RooArgSet(x), RooFit.Import(tree1))

#    # TODO: export somewhere? does not need to be defined inside...
#    mean = RooRealVar("mean", "Mean of Double CB PDF", 5280, 5100, 5600)#, 5300, 5500)
#    sigma = RooRealVar("sigma", "Sigma of Double CB PDF", 40, 0.001, 200)
#    alpha_0 = RooRealVar("alpha_0", "alpha_0 of one side", 5.715)#, 0, 150)
#    alpha_1 = RooRealVar("alpha_1", "alpha_1 of other side", -4.019)#, -200, 0.)
#    lambda_0 = RooRealVar("lambda_0", "Exponent of one side", 3.42)#, 0, 150)
#    lambda_1 = RooRealVar("lambda_1", "Exponent of other side", 3.7914)#, 0, 500)
#
#    # TODO: export somewhere? pdf construction
#    frac = RooRealVar("frac", "Fraction of crystal ball pdfs", 0.479, 0.01, 0.99)
#
#    crystalball1 = RooCBShape("crystallball1", "First CrystalBall PDF", x,
#                              mean, sigma, alpha_0, lambda_0)
#    crystalball2 = RooCBShape("crystallball2", "Second CrystalBall PDF", x,
#                              mean, sigma, alpha_1, lambda_1)
#    doubleCB = RooAddPdf("doubleCB", "Double CrystalBall PDF",
#                         crystalball1, crystalball2, frac)

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
        n_sig = RooRealVar("n_sig", "Number of signal events", 1050, 0, 200000)

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

#    # create bkg-pdf
#    lambda_exp = RooRealVar("lambda_exp", "lambda exp pdf bkg", -0.00025, -1., 1.)
#    bkg_pdf = RooExponential("bkg_pdf", "Background PDF exp", x, lambda_exp)

    if blind:
        comb_pdf = RooAddPdf("comb_pdf", "Combined DoubleCB and bkg PDF",
                             RooArgList(sig_pdf, bkg_pdf), RooArgList(blind_n_sig, n_bkg))
    else:
        comb_pdf = RooAddPdf("comb_pdf", "Combined DoubleCB and bkg PDF",
                             RooArgList(sig_pdf, bkg_pdf), RooArgList(n_sig, n_bkg))

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
#    print "n_cpu:", meta_config.get_n_cpu()
#    input("test")
#    comb_pdf.fitTo(data, RooFit.Extended(ROOT.kTRUE), RooFit.NumCPU(meta_config.get_n_cpu()))
#     HACK to get 8 cores in testing
    c5 = TCanvas("c5", "RooFit pdf not fit vs " + data_name)
    c5.cd()
    x_frame1 = x.frame()
#    data.plotOn(x_frame1)
#    comb_pdf.pdfList()[1].plotOn(x_frame1)

    if __name__ == "__main__":
        n_cpu = 8
    else:
        n_cpu = meta_config.get_n_cpu()
        print "n_cpu = ", n_cpu
        # HACK
#        n_cpu = 8
    result_fit = comb_pdf.fitTo(data, RooFit.Minos(ROOT.kTRUE),
                                RooFit.Extended(ROOT.kTRUE),
                                RooFit.NumCPU(n_cpu))
    # HACK end
    if bkg_in_region:
        x.setRange("signal", bkg_in_region[0], bkg_in_region[1])
        bkg_pdf_fitted = comb_pdf.pdfList()[1]
        int_argset = RooArgSet(x)
#        int_argset = x
#        int_argset.setRange("signal", bkg_in_region[0], bkg_in_region[1])
        integral = bkg_pdf_fitted.createIntegral(int_argset,
                                                 RooFit.NormSet(int_argset),
                                                 RooFit.Range("signal"))
        bkg_cdf = bkg_pdf_fitted.createCdf(int_argset, RooFit.Range("signal"))
        bkg_cdf.plotOn(x_frame1)


#        integral.plotOn(x_frame1)
        n_bkg_below_sig = integral.getVal(int_argset) * n_bkg.getVal()
        x_frame1.Draw()

    if plot_importance >= 3:
        c2 = TCanvas("c2", "RooFit pdf fit vs " + data_name)
        c2.cd()
        x_frame = x.frame()
#        if log_plot:
#            c2.SetLogy()
#        x_frame.SetTitle("RooFit pdf vs " + data_name)
        x_frame.SetTitle(data_name)
        if pulls:
            pad_data = ROOT.TPad("pad_data", "Pad with data and fit", 0, 0.33, 1, 1)
            pad_pulls = ROOT.TPad("pad_pulls", "Pad with data and fit", 0, 0, 1, 0.33)
            pad_data.SetBottomMargin(0.00001)
            pad_data.SetBorderMode(0)
            if log_plot:
                pad_data.SetLogy()
            pad_pulls.SetTopMargin(0.00001)
            pad_pulls.SetBottomMargin(0.2)
            pad_pulls.SetBorderMode(0)
            pad_data.Draw()
            pad_pulls.Draw()
            pad_data.cd()
        else:
            if log_plot:
                c2.SetLogy()
    if blind:
        # HACK
        column = 'x'
        # END HACK
        x.setRange("lower", min_x, lower_blind)
        x.setRange("upper", upper_blind, max_x)
        range_str = "lower,upper"
        lower_cut_str = str(min_x) + "<=" + column + "&&" + column + "<=" + str(lower_blind)
        upper_cut_str = str(upper_blind) + "<=" + column + "&&" + column + "<=" + str(max_x)
        sideband_cut_str = "("+lower_cut_str+")" + "||" + "("+upper_cut_str+")"

        n_entries = data.reduce(sideband_cut_str).numEntries() / data.numEntries()
#        raw_input("n_entries: " + str(n_entries))
        if plot_importance >= 3:
            data.plotOn(x_frame, RooFit.CutRange(range_str), RooFit.NormRange(range_str))
            comb_pdf.plotOn(x_frame, RooFit.Range(range_str),
                            RooFit.Normalization(n_entries, RooAbsReal.Relative),
                            RooFit.NormRange(range_str))
            if pulls:
#                pull_hist(pull_frame=x_frame, pad_data=pad_data, pad_pulls=pad_pulls)
                x_frame_pullhist = x_frame.pullHist()
    else:
        if plot_importance >= 3:
            data.plotOn(x_frame)
            comb_pdf.plotOn(x_frame)
            if pulls:
                pad_pulls.cd()
                x_frame_pullhist = x_frame.pullHist()
                pad_data.cd()

            comb_pdf.plotOn(x_frame,
                            RooFit.Components(sig_pdf.namePtr().GetName()),
                            RooFit.LineStyle(ROOT.kDashed))
            comb_pdf.plotOn(x_frame,
                            RooFit.Components(bkg_pdf.namePtr().GetName()),
                            RooFit.LineStyle(ROOT.kDotted))
#            comb_pdf.plotPull(n_sig)



    if plot_importance >= 3:
        x_frame.Draw()

        if pulls:
            pad_pulls.cd()
            x_frame.SetTitleSize(0.05, 'Y')
            x_frame.SetTitleOffset(0.7, 'Y')
            x_frame.SetLabelSize(0.04, 'Y')

#            c11 = TCanvas("c11", "RooFit\ pulls" + data_name)
#            c11.cd()
#            frame_tmp = x_frame
            frame_tmp = x.frame()

#            frame_tmp.SetTitle("significance")

            frame_tmp.SetTitle("Roofit\ pulls\ " + data_name)
            frame_tmp.addObject(x_frame_pullhist)

            frame_tmp.SetMinimum(-5)
            frame_tmp.SetMaximum(5)

#            frame_tmp.GetYaxis().SetTitle("significance")
            frame_tmp.GetYaxis().SetNdivisions(5)
            frame_tmp.SetTitleSize(0.1, 'X')
            frame_tmp.SetTitleOffset(1, 'X')
            frame_tmp.SetLabelSize(0.1, 'X')
            frame_tmp.SetTitleSize(0.1, 'Y')
            frame_tmp.SetTitleOffset(0.5, 'Y')
            frame_tmp.SetLabelSize(0.1, 'Y')

            frame_tmp.Draw()


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

#        input("press ENTER to show plot")
        sframe.Draw()

    if plot_importance >= 3:
        pass

    params = comb_pdf.getVariables()
    params.Print("v")

#    print bkg_cdf.getVal()

    if sPlot:
        sPlotData = ROOT.RooStats.SPlot("sPlotData","sPlotData",
                        data,  # variable fitted to, RooDataSet
                        comb_pdf,  # fitted pdf
                        ROOT.RooArgList(n_sig,
                                        n_bkg,
#                                                NSigB0s
                                        ))
        sweights = np.array([sPlotData.GetSWeight(i, 'n_sig') for i in range(data.numEntries())])
        return n_sig.getVal(), n_bkg_below_sig, sweights

    if blind:
        return blind_n_sig.getVal(), n_bkg_below_sig, comb_pdf
    else:
        return n_sig.getVal(), n_bkg_below_sig, comb_pdf


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

def pull_hist(pull_frame, pad_data, pad_pulls):
    """Add pulls into the current pad."""
    pad_data.cd()
    dataHist = pull_frame.getHist("datahistogram")
    curve1 = pull_frame.getObject(1)  # 1 is index in the list of RooPlot items (see printout from massplot->Print("V")
    curve2 = pull_frame.getObject(2)
    hresid1 = dataHist.makePullHist(curve1, True)
    hresid2 = dataHist.makePullHist(curve2, True)

    # RooHist* hresid = massplot->pullHist("datahistogram","blindtot")
    pad_pulls.cd()
#    resid = M_OS.frame()
    pull_frame.addPlotable(hresid1,"P")
    pull_frame.addPlotable(hresid2,"P")
    pull_frame.SetTitle("")
#    pull_frame.GetXaxis().SetTitle("#it{m}(#it{#pi}^{ #plus}#it{#pi}^{ #minus}) [MeV/#it{c}^{2}]")
#    gStyle->SetPadLeftMargin(0.1)




def metric_vs_cut_fitted(data, predict_col, fit_col, sig_pdf, bkg_pdf, x, region,
                         second_storage=None, metric='punzi',
                         n_sig=None, n_bkg=None, stepsize=0.025,
                         plot_importance=3):
    """Calculate a metric vs a given cut by estimating the bkg from the fit.

    Parameters
    ----------
    data : HEPDataStorage

    predict_col : str

    fit_col : str

    region : tuple(numerical, numerical)
        The lower and upper points to integrate over.
    x : RooRealVar

    """
    from raredecay.tools.metrics import punzi_fom, precision_measure

    metric_name = metric
    if metric == 'punzi':
        metric = punzi_fom
    elif metric == 'precision':
        metric = precision_measure
    # TODO: convert meric strings to metric
    n_steps = int(np.floor_divide(1, stepsize))
    if n_steps < 1:
        raise ValueError("stepsize has to be smaller then 1, not", stepsize)
    cuts = np.linspace(0, 1, num=n_steps, endpoint=False)
    plots = int(10 / n_steps)
    current_plot = 0

    if not type(predict_col) == type(fit_col) == str:
        raise TypeError("predict_col and/or fit_col is not a string but has to be.")

    scores = []
    debug1 = []
    for cut in cuts:

        if plot_importance > 2:
            temp_plot_importance = plot_importance if plots > current_plot else 0

        temp_data = data.copy_storage(columns=[predict_col, fit_col], add_to_name="")
        temp_df = temp_data.pandasDF()
        temp_df = temp_df[cut < temp_df[predict_col]]
        temp_data.set_data(temp_df)

        n_sig_weighted = sum(temp_data.get_weights()[temp_data.get_targets() == 1])
        if second_storage is not None:

            temp_second_storage = second_storage.copy_storage(columns=[predict_col, fit_col],
                                                              add_to_name="")
            temp_df = temp_second_storage.pandasDF()
            temp_df = temp_df[cut < temp_df[predict_col]]
            temp_second_storage.set_data(temp_df)
            n_sig_weighted += sum(temp_second_storage.get_weights()[temp_second_storage.get_targets() == 1])
        else:
            temp_second_storage = second_storage


        n_sig_fit, n_bkg_fit = fit_mass(data=temp_data, column=fit_col, x=x, sig_pdf=sig_pdf,
                                bkg_pdf=bkg_pdf, n_sig=n_sig, n_bkg=n_bkg, blind=False,
                                nll_profile=False, second_storage=temp_second_storage,
                                plot_importance=temp_plot_importance,
                                bkg_in_region=region)

        scores.append(metric(n_signal=n_sig_weighted, n_background=n_bkg_fit))
        debug1.append({'n_sig': n_sig_fit, 'n_bkg': n_bkg_fit, 'n_sig_weighted': n_sig_weighted})

        # DEBUG
        import time
        print "scores:", scores
        print "debug1:", debug1
#        time.sleep(8)

    print debug1

    return cuts, scores



if __name__ == '__main__':

#    data = RooDataSet("data", )
    from raredecay.tools.data_storage import HEPDataStorage
    import pandas as pd
    import matplotlib.pyplot as plt

#    np.random.seed(40)

#    mode = "fit"
#    mode = 'fit_metric'
    mode = "sPlot"

# create signal pdf BEGIN
    lower_bound = 4800
#    lower_bound = 5000
    x = RooRealVar("x", "x variable", lower_bound, 6000)

#    x = RooRealVar("x", "x variable", 4800, 6000)

    # TODO: export somewhere? does not need to be defined inside...
    mean = RooRealVar("mean", "Mean of Double CB PDF", 5280, 5270, 5290)#, 5300, 5500)
    sigma = RooRealVar("sigma", "Sigma of Double CB PDF", 40, 0, 45)
    alpha_0 = RooRealVar("alpha_0", "alpha_0 of one side", 40, 30, 50)
    alpha_1 = RooRealVar("alpha_1", "alpha_1 of other side", -40, -50, -30.)
    lambda_0 = RooRealVar("lambda_0", "Exponent of one side", 40, 30, 50)
    lambda_1 = RooRealVar("lambda_1", "Exponent of other side", 40, 30, 50)

    # TODO: export somewhere? pdf construction
    frac = RooRealVar("frac", "Fraction of crystal ball pdfs", 0.479, 0.01, 0.99)

    crystalball1 = RooCBShape("crystallball1", "First CrystalBall PDF", x,
                              mean, sigma, alpha_0, lambda_0)
    crystalball2 = RooCBShape("crystallball2", "Second CrystalBall PDF", x,
                              mean, sigma, alpha_1, lambda_1)
    doubleCB = RooAddPdf("doubleCB", "Double CrystalBall PDF",
                         crystalball1, crystalball2, frac)
# create signal pdf END

    # create bkg-pdf BEGIN
    lambda_exp = RooRealVar("lambda_exp", "lambda exp pdf bkg", -0.002, -10., -0.000001)
    bkg_pdf = RooExponential("bkg_pdf", "Background PDF exp", x, lambda_exp)
    # create bkg-pdf END

    n_sig = 2500

    data = pd.DataFrame(np.random.normal(loc=5280, scale=37, size=(n_sig, 3)), columns=['x', 'y', 'pred'])
#    data['pred'] = np.array([min((abs(y), 0.99)) for y in np.random.normal(loc=0.6, scale=0.25, size=n_sig)])
    bkg_data = np.array([i for i in (np.random.exponential(scale=300,
                                     size=(7500, 3)) + 4800) if i[0] < 6000])
    bkg_data[:, 2] = np.array([min((abs(y), 0.96)) for y in np.random.normal(loc=0.4,
                             scale=0.4, size=len(bkg_data))])
    data = pd.concat([data, pd.DataFrame(bkg_data, columns=['x', 'y', 'pred'])], ignore_index=True)

    data = HEPDataStorage(data, target=np.concatenate((np.ones(n_sig),
                                                       np.zeros(len(bkg_data)))))
    data_copy = data.copy_storage()

    if mode == 'fit':
        fit_result = fit_mass(data=data, column='x', sig_pdf=doubleCB, x=x,
                                       bkg_pdf=bkg_pdf,
                                       blind=False,
#                                       blind=(5100, 5380),
                                       plot_importance=4, #bkg_in_region=(5100, 5380)
                                       )
        print fit_result
        print "True values: nsig =", n_sig, " n_bkg =", len(bkg_data)

    elif mode == 'fit_metric':
        result = metric_vs_cut_fitted(data=data, predict_col='pred', fit_col='x',
                                      sig_pdf=doubleCB, bkg_pdf=bkg_pdf, x=x,
                                      region=(5100, 5380), stepsize=0.01)
        print result

        plt.plot(*result)


    elif mode == 'sPlot':
        fit_result = fit_mass(data=data, column='x', sig_pdf=doubleCB, x=x,
                                       bkg_pdf=bkg_pdf,
                                       blind=False,
                                       plot_importance=1, #bkg_in_region=(5100, 5380)
                                       sPlot=True
                                       )
        n_sig, n_bkg, sweights = fit_result
        import copy
        sweights = copy.deepcopy(sweights)
        plt.figure("new figure")
#        plt.hist(range(100))
#        plt.figure("new figure")
        plt.hist(sweights, bins=30)

        data_copy.set_weights(sweights)
        data_copy.plot()


#    n_sig_fit.append(np.mean(n_sig_averaged))#[5300, 5500]))
#    print n_sig_fit
#    raw_input("hiii")
#    plt.plot(np.array(n_sig_gen), np.array(n_sig_fit), linestyle='--', marker='o')
#    plt.plot(n_sig_gen, n_sig_gen, linestyle='-')
    plt.show()


    print "finished"
