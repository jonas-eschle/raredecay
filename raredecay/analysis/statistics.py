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
#from ROOT import TFile, TLorentzVector, TVector3, TRotation, TLorentzRotation, TMath, TH1D, TCanvas, TH2D, TObject, TF1, TH1F, gStyle, TF2, TF3, TF12, TFormula
#from ROOT import RooRealVar, RooFormulaVar, RooArgList, RooArgSet, RooLegendre, RooProdPdf, RooPolynomial, RooAddPdf, RooPlot, RooProduct, RooDataSet, RooKeysPdf
#from ROOT import RooFit
from itertools import repeat
import itertools as it
import csv
import re
import sqlite3
import os.path
import numpy as np
from array import array

import sys

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
    print "hello world"
    ROOT.TFile()