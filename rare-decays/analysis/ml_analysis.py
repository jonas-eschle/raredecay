# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: mayou
"""

import hep_ml

from tools import dev_tool
import config as cfg

class MachineLearningAnalysis:
    """Use ML-techniques on datasets to reweight, train and classify


    """

    __REWEIGHT_MODE = {'gb': 'GB', 'bin': 'Bins'}  # for GB/BinsReweighter
    __REWEIGHT_MODE_DEFAULT = 'gb'  # user-readable

    def __init__(self):
        self.logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)

    def reweight_mc_real(self, bin_or_gb='gb'):
        if bin_or_gb not in self.__REWEIGHT_MODE:
            self.logger.warning(str(bin_or_gb) + " not a valid choice of " +
                                str(self.__REWEIGHT_MODE.keys()) +
                                ". Instead, the default value was used: " +
                                self.__REWEIGHT_MODE_DEFAULT)
            bin_or_gb = self.__REWEIGHT_MODE_DEFAULT
        bin_or_gb = self.__REWEIGHT_MODE.get(bin_or_gb)


        self.logger.info("module finished")
