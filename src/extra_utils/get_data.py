"""
Load MovieLens small dataset.

Author: Liam Schoneveld
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import zipfile
import pandas as pd
import numpy as np


DEFAULT_SOURCE_URL = \
    'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'


def _make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _maybe_download(filename, directory, source_url):
    _make_dir_if_not_exist(directory)
    zippath = os.path.join(directory, filename)
    print(zippath)
    checkpath = os.path.join(directory, 'ml-latest-small')
    if not os.path.exists(checkpath):
        print('Downloading', filename, '...')
        filepath, _ = urllib.request.urlretrieve(source_url, zippath)
        print('Successfully downloaded', filename)
        f = zipfile.ZipFile(zippath, 'r')
        f.extractall(directory)
        os.remove(zippath)
    return checkpath


def load_movielens_small(data_dir):
    
    # Download the dataset if we don't have it already
    path = _maybe_download('ml-latest-small.zip', data_dir, DEFAULT_SOURCE_URL)
    
    # Load the dataset
    data = {}
    for f in ['links', 'movies', 'ratings', 'tags']:
        data[f] = pd.read_csv(os.path.join(path, f + '.csv'))
        
    table_names = ['links', 'movies', 'ratings', 'tags']
    links, movies, ratings, tags = [data[i] for i in table_names]

    for i in table_names:
        # Make user IDs zero-indexed
        for col in ['userId', 'movieId']:
            if col in data[i].columns:
                data[i][col] = data[i][col] - 1
        print(i)
        print(data[i].head())
        
    ratings['rating'] = ratings['rating'].astype(np.float32)
    
    return data, table_names, links, movies, ratings, tags
