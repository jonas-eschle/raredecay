# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 20:00:33 2016

@author: mayou
"""

'''
Basic script to estimate the number of background to be used in the different FoM's for the B->Kst tau tau optimisation
'''

import sys
import argparse
import ROOT
from ROOT import TFile, TLorentzVector, TVector3, TRotation, TLorentzRotation, TMath, TH1D, TCanvas, TH2D, TObject, TF1, TH1F, gStyle, TF2, TF3, TF12, TFormula
from ROOT import RooRealVar, RooFormulaVar, RooArgList, RooArgSet, RooLegendre, RooProdPdf, RooPolynomial, RooAddPdf, RooPlot, RooProduct, RooDataSet, RooKeysPdf
from ROOT import RooFit, RooCBShape, RooGaussian
from itertools import repeat
import itertools as it
import csv
import re
import sqlite3
import os.path
import numpy as np
from array import array

import sys

def fit_mass(data, column='B_M'):


    mean = RooRealVar("mean", "Mean of Double CB PDF", 5366, 5000, 6000)
    sigma = RooRealVar("sigma", "Sigma of Double CB PDF", 8, 0, 25)
    alpha_0 = RooRealVar("alpha_0", "alpha_0 of one side", 1, 0, 5)
    alpha_1 = RooRealVar("alpha_1", "alpha_1 of other side", -1, 0, -5)
    lambda_0 = RooRealVar("lambda_0", "Exponent of one side", 1, 0, 5)
    lambda_1 = RooRealVar("lambda_1", "Exponent of other side", 1, 0, 5)

#    RooCBShape cb_1




#        fitter.makeDoubleCB(((*it_modes)+"_"+(*it_year)+"_"+(*it_trig)+"_"+(*it_nBrem)+"_"+(*it_bin)+"_pdf").c_str(),
#                      //start   min     max
#                        5366,   5356,   5376,   // mu; most probable value, resonance mass
#                        8,      0,      25,     // sigma; resolution
#                        1,      0,      5,      // alpha_0; transition point
#                        -1,     -5,     0,      // alpha_0 other side;
#                        1,      0,      5,      // exponent;
#                        1,      0,      5,      // exponent other side;
#                        0.5);
#     }


def fitBMassReco(File , Tree , splitVal , cutVariable ):
#   print "calculating best cut for proba= " , splitVal
   # Open Monte Carlo dataset for calculating the efficiency
#   f = TFile('/home/hep/rsilvaco/Analysis/RareDecays/Bd2Ksttautau/Ntuples/MonteCarlo/2011/B2KstTauTau-MonteCarlo-2011-MagDown-Training.root', 'read')
   f = TFile(File, 'read')
   t = f.Get(Tree)

   cutVal_Name = cutVariable[1:]
#   print cutVal_Name
   cut = str(splitVal) + cutVariable
   B_M_calc_Nom_mpi_sel = RooRealVar('B_M_calc_Nom_mpi_sel','Hyp 1', 4000, 40000, 'MeV/c^{2}')
   proba_GB = RooRealVar(cutVal_Name,'Probability GB', 0, 1)
#   proba_GB = RooRealVar('proba_GB','Probability GB', 0, 1)
   sigDataset = RooDataSet('sigDataset','Signal Dataset', RooArgSet(B_M_calc_Nom_mpi_sel, proba_GB), RooFit.Import(t), RooFit.Cut(cut))

   poly = RooKeysPdf("poly","Polynomial function", B_M_calc_Nom_mpi_sel, sigDataset)
   integral = poly.createIntegral(RooArgSet(B_M_calc_Nom_mpi_sel))

   # Find the optimal
   integral_sig = None
   intLimit = None
   minLimit = 4900
   for iVal in range(minLimit, 20000, 50):
#      print iVal
      B_M_calc_Nom_mpi_sel.setRange("signal",4800,iVal)
      integral_sig = poly.createIntegral(RooArgSet(B_M_calc_Nom_mpi_sel), RooFit.Range("signal"))
      if (integral_sig.getValV() > 0.680 ):# and ( integral_sig.getValV() < 0.684 ):
#         print splitVal, ' \t : ', iVal
         return  4800, iVal
   print  splitVal, ' \t : ', 14000 ,'  Out of bounds'
   return 4800, 14000

if __name__ == '__main__':
    mean = RooRealVar("mean", "Mean of Gaussian", 0)
    sigma = RooRealVar("sigma", "Width of Gaussian", 2)
    x = RooRealVar("x", "x", -20, 20)
    gauss1 = RooGaussian("gauss1", "Gaussian test dist", x, mean, sigma)
    xframe = x.frame()
    gauss1.plotOn(xframe)
    xframe.Draw()
    print gauss1.generate(x, 10000)
#    data = RooDataSet("data", "Gaussian generated data-set", gauss1.generate(x, 10000))