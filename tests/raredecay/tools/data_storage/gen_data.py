# Python 2 backwards compatibility overhead START
from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct,
                      open, pow, range, round, str, super, zip)
import sys
import warnings
import raredecay.meta_config

try:
    from future.builtins.disabled import (apply, cmp, coerce, execfile, file, long, raw_input,
                                          reduce, reload, unicode, xrange, StandardError)
    from future.standard_library import install_aliases

    install_aliases()
except ImportError as err:
    if sys.version_info[0] < 3:
            raise err
# Python 2 backwards compatibility overhead END


if __name__ == '__main__':
    import pandas as pd
    from root_pandas import to_root

    from test_HEPDataStorage import TestHEPDataStorageMixin
    
    data, targets, weights, index = TestHEPDataStorageMixin._make_dataset()

    data['root_w'] = weights
    # data.reset_index(inplace=True, drop=True)
    to_root(data.loc[data.index], 'ds1.root', key='DecayTree')
