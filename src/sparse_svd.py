from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np


def fit_and_predict_with_svd(train_ratings, test_ratings, num_users,
                             num_movies_with_ratings):
    user_movie_rating = train_ratings[['userId', 'movieId', 'rating']].values
    row, col, data = [user_movie_rating[:, i] for i in range(user_movie_rating.shape[1])]
    
    train_mat = csr_matrix((data, (row, col)), 
                           shape=[num_users, num_movies_with_ratings])
    u, s, vt = svds(train_mat)

    
    svd_preds = np.dot(np.dot(u, np.diag(s)), vt)
    svd_preds_train = svd_preds[train_ratings['userId'].values,
                                train_ratings['movieId'].values]
    svd_preds_test = svd_preds[test_ratings['userId'].values,
                               test_ratings['movieId'].values]
    
    return svd_preds_train, svd_preds_test
