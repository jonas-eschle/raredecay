from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals


if __name__ == '__main__':
    import pandas as pd
    from root_pandas import to_root

    from test_HEPDataStorage import TestHEPDataStorageMixin
    
    data, targets, weights, index = TestHEPDataStorageMixin._make_dataset()

    data['root_w'] = weights

    to_root(data.loc[data.index], 'ds1.root', key='DecayTree')
