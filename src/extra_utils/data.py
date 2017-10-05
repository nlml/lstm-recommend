"""
Load MovieLens small dataset.

Author: Liam Schoneveld
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def crop_to_most_frequent(ratings, pct_to_keep=20, colname='movieId'):
    movie_counts = ratings.groupby(colname).apply(len)
    pct_to_remove = 100 - pct_to_keep
    movies_to_keep = movie_counts[movie_counts >= np.percentile(movie_counts, pct_to_remove)].index.values
    keep_rows = ratings[colname].isin(movies_to_keep)
    print('Shape before drop less popular movies {}'.format(ratings.shape))
    ratings = ratings[keep_rows]
    print('Shape after drop less popular movies {}'.format(ratings.shape))
    return ratings