"""
Other useful functions.

Author: Liam Schoneveld
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def index_of_each_y_in_x(x, y):
    """For each element in a np.array y, return its position in x"""
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    return xsorted[ypos]