"""
Author: Liam Schoneveld
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def get_cv_fold_idxs(n, num_folds, rng=None, shuffle=False):
    if shuffle:
        assert rng, 'Must provide rng=np.random.RandomState when shuffle=True'
        idxs = rng.permutation(n)
    else:
        idxs = np.arange(n)
    slice_sizes = n * 1. / num_folds
    idxs_train, idxs_test = [], []
    for f in range(num_folds):
        start = int(f * slice_sizes)
        if f == (num_folds - 1):
            end = len(idxs)
        else:
            end = int((f + 1) * slice_sizes)
        idxs_test.append(idxs[start:end].copy())
        idxs_train.append(idxs.copy())
        idxs_train[-1] = np.delete(idxs_train[-1], idxs_test[-1])
    return idxs_train, idxs_test
