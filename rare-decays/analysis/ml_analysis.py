# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:29:01 2016

@author: mayou
"""

import hep_ml.reweight
from tools import dev_tool, data_tools
import config as cfg


class MachineLearningAnalysis:
    """Use ML-techniques on datasets to reweight, train and classify


    """

    __REWEIGHT_MODE = {'gb': 'GB', 'bins': 'Bins'}  # for GB/BinsReweighter
    __REWEIGHT_MODE_DEFAULT = 'gb'  # user-readable

    def __init__(self):
        self.logger = dev_tool.make_logger(__name__, **cfg.logger_cfg)

    def reweight_mc_real(self, reweight_data_mc, reweight_data_real,
                         reweighter='gb', weights_real=None,
                         reweight_tree_mc=None, reweight_tree_real=None,
                         branch_names=None, reweight_saveas=None,
                         meta_cfg=None):
        """Return weight from a mc/real comparison.
        """
        try:
            reweighter = self.__REWEIGHT_MODE.get(reweighter)
        except KeyError:
            self.logger.critical("Reweighter invalid: " + reweighter +
                                 ". Probably wrong defined in config.")
            raise ValueError
        else:
            reweighter += 'Reweighter'


        self.logger.debug("starting data conversion")
        original = data_tools.to_pandas(reweight_data_mc,
                                        tree=reweight_tree_mc,
                                        columns=branch_names)
        target = data_tools.to_pandas(reweight_data_real,
                                      tree=reweight_tree_real,
                                      columns=branch_names)
        self.logger.debug("data converted to pandas")
        reweighter = getattr(hep_ml.reweight,
                             reweighter)(**meta_cfg)
        reweighter.fit(original, target)
        return data_tools.adv_return(reweighter, self.logger,
                                     save_name=reweight_saveas)


    def reweight_weights(self, reweight_apply_data, reweighter_trained):
        """ Return the new weights by applying a given reweighter
        """
        reweighter_trained = data_tools.try_unpickle(reweighter_trained)
        reweight_apply_data = data_tools.to_pandas(reweight_apply_data)
        new_weights = reweighter_trained.predict_weights(reweight_apply_data)
        return new_weights


        self.logger.info("module finished")
