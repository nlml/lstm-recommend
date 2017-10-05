"""
"""
import numpy as np
from extra_utils.get_cv_fold_idxs import get_cv_fold_idxs
from extra_utils.get_data import load_movielens_small
from sparse_svd import fit_and_predict_with_svd
from extra_utils.other import index_of_each_y_in_x
from extra_utils.data import crop_to_most_frequent
from nn import BasicNN


DATA_DIR = '../data'
SEED = 1
CV_FOLDS = 5
PCT_OF_MOVIES_TO_KEEP = 80
PCT_OF_USERS_TO_KEEP = 100

rng = np.random.RandomState(SEED)

# Load data
data, tbl_names, links, movies, ratings, tags = load_movielens_small(DATA_DIR)

# Remove movies with not many ratings
ratings = crop_to_most_frequent(ratings, pct_to_keep=PCT_OF_MOVIES_TO_KEEP)
ratings = crop_to_most_frequent(ratings, pct_to_keep=PCT_OF_USERS_TO_KEEP, 
                                colname='userId')

# Print some statistics of the data
movies_with_ratings = np.array(list(sorted(ratings['movieId'].unique())))
users_with_ratings = np.array(list(sorted(ratings['userId'].unique())))
num_movies_with_ratings = len(movies_with_ratings)
num_users_with_ratings = len(users_with_ratings)
print('Number of movies with ratings: {}'.format(num_movies_with_ratings))
print('Number of users with ratings: {}'.format(num_users_with_ratings))

# Make training and testing CV folds
idxs_train, idxs_test = get_cv_fold_idxs(
        ratings.shape[0], CV_FOLDS, rng, shuffle=False)

# Make new movie IDs, starting at 0, only including movies with ratings
ratings['movieIdOrig'] = ratings['movieId'].copy()
ratings['userIdOrig'] = ratings['userId'].copy()
ratings['movieId'] = index_of_each_y_in_x(
    movies_with_ratings, ratings['movieId'].values)
ratings['userId'] = index_of_each_y_in_x(
    users_with_ratings, ratings['userId'].values)

    
mse = lambda a, b: np.mean(np.square(a - b))

for fold in range(CV_FOLDS):
    print('\nFold {} of {}'.format(fold+1, CV_FOLDS))
    train_ratings = ratings.iloc[idxs_train[fold]]
    test_ratings = ratings.iloc[idxs_test[fold]]
    
    print('\nMean:')
    print('Test MSE: %.4f' % mse(
            train_ratings['rating'].mean(), test_ratings['rating'].values))
    
    print('\nMedian:')
    print('Test MSE: %.4f' % mse(
            train_ratings['rating'].median(), test_ratings['rating'].values))
    
# =============================================================================
#     print('\nSVD:')
#         
#     svd_preds_train, svd_preds_test = fit_and_predict_with_svd(
#             train_ratings, test_ratings, num_users, num_movies_with_ratings)
#     
#     print('Train MSE: %.4f' % mse(
#             svd_preds_train, train_ratings['rating'].values))
#     print(' Test MSE: %.4f' % mse(
#             svd_preds_test, test_ratings['rating'].values))
# =============================================================================
    
    print('\nNN:')
    nn = BasicNN(rng, num_users_with_ratings, num_movies_with_ratings, emb_dim=10,
                 batch_size=32, l2_reg=0.01, emb_reg=1., init_std=0.1)
    nn.train(train_ratings, test_ratings, num_iters=500000, eval_times=50)
    
    cols = ['userId', 'movieIdZeroIndexed']
    nn_preds_train = nn.predict(*list(train_ratings[cols].values.T))
    nn_preds_test = nn.predict(*list(test_ratings[cols].values.T))
    print('Train MSE: %.4f' % mse(
            nn_preds_train, train_ratings['rating'].values))
    print(' Test MSE: %.4f' % mse(
            nn_preds_test, test_ratings['rating'].values))





