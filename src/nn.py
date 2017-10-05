import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim


MAX_RATING = 5.0
MIN_RATING = 0.5


class DataGen(object):
    
    def __init__(self, rng, ratings):
        self.ratings = ratings
        self.rng = rng
        self.num_examples = len(self.ratings)
        self.fixed_perm = self.rng.permutation(self.num_examples)
        self.reset_perm(fixed=True)

    def reset_perm(self, fixed=False):
        """
        If we reach the end of the dataset, we need to make a new random order.

        :param fixed: bool; whether to use the same ordering set in __init__()
        :return: None
        """
        if fixed:
            self.perm = self.fixed_perm
            self.curr_idx = 0
        else:
            self.perm = self.rng.permutation(self.num_examples)
            self.curr_idx = 0
        
    def next_batch(self, batch_size, reset_after=False):
        if (self.curr_idx + batch_size) > self.num_examples:
            self.reset_perm()
        st = self.curr_idx
        en = self.curr_idx + batch_size
        
        
        sel_rows = self.ratings.iloc[self.perm[st:en], :]
        values_user_movie = sel_rows[['userId', 'movieId']].values
        values_rating = sel_rows['rating']
        user_idxs, movie_idxs = list(values_user_movie.T)
        
        if reset_after:
            self.curr_idx = 0
        else:
            self.curr_idx = en
        
        return user_idxs, movie_idxs, values_rating


class BasicNN(object):


    def __init__(self, rng, num_users, num_movies_with_ratings, emb_dim=10,
                 batch_size=32, l2_reg=0.01, emb_reg=1.0, init_std=0.1):
        self.num_users = num_users
        self.num_movies_with_ratings = num_movies_with_ratings
        self.rng = rng
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.emb_reg = emb_reg
        self.l2_reg = l2_reg
        self.init_std = init_std
        self.build_graph()


    def make_placeholders(self):
    
        self.user_idxs_in = tf.placeholder(tf.int32, [None])
        self.movie_idxs_in = tf.placeholder(tf.int32, [None])
        self.targets_in = tf.placeholder(tf.float32, [None])
        
        
    def build_graph(self):
        tf.reset_default_graph()
        self.make_placeholders()
        self.forward()
        self.make_losses()
        self.make_train_op()
        
        
    def forward(self):
        self.user_embs = tf.Variable(self.rng.normal(
                0, self.init_std, [self.num_users, self.emb_dim]).astype(np.float32))
        self.movie_embs = tf.Variable(self.rng.normal(
                0, self.init_std,
                [self.num_movies_with_ratings, self.emb_dim]).astype(np.float32))
        
        self.user_embs_batch = tf.nn.embedding_lookup(
                self.user_embs, self.user_idxs_in)
        self.movie_embs_batch = tf.nn.embedding_lookup(
                self.movie_embs, self.movie_idxs_in)
        
        
        self.concat = tf.concat(
                [self.user_embs_batch, self.movie_embs_batch], axis=1)
        rg = slim.l2_regularizer(self.l2_reg)
        self.preds = slim.fully_connected(self.concat, 1,
                                          weights_regularizer=rg)
        
        #range_of_ratings_vals = MAX_RATING - MIN_RATING
        #self.dot_prod = tf.reduce_sum(
        #        self.user_embs_batch * self.movie_embs_batch, axis=1)
        #self.preds = MIN_RATING + range_of_ratings_vals * tf.nn.sigmoid(
        #                                                        self.dot_prod)
        
        
    def make_losses(self):
        self.losses = {}
        self.losses['l2_user'] = tf.reduce_mean(
                tf.square(self.user_embs_batch)) * self.emb_reg
        self.losses['l2_movie'] = tf.reduce_mean(
                tf.square(self.movie_embs_batch)) * self.emb_reg
        self.losses['l2'] = tf.add_n(tf.losses.get_regularization_losses())
        self.losses['mse'] = tf.reduce_mean(tf.squared_difference(
                self.preds, self.targets_in))
        self.losses['total'] = tf.add_n([v for k, v in self.losses.items()])
        
        
    def make_train_op(self):
        self.train_op = tf.train.AdamOptimizer().minimize(self.losses['total'])


    def get_feed(self, data_gen, batch_size, reset=False):
        user_idxs, movie_idxs, ratings_vals = data_gen.next_batch(
                batch_size, reset_after=reset)
        return {self.user_idxs_in: user_idxs, 
                self.movie_idxs_in: movie_idxs, 
                self.targets_in: ratings_vals}
    
    
    def do_eval(self, sess, it):
        if it > 0:
            print('Train loss mean: %.4f' % (self.loss_sum / self.eval_freq))
        print('Test loss: %.4f' % self.sess.run(self.losses['mse'], 
            self.get_feed(self.data_gen_test, 
                          self.data_gen_test.num_examples, reset=True)))
        self.loss_sum = 0.
        

    def train(self, train_ratings, test_ratings, num_iters=10000, 
              eval_times=5):
        
        self.data_gen_train = DataGen(self.rng, train_ratings)
        self.data_gen_test = DataGen(self.rng, test_ratings)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.loss_sum = 0.
        self.eval_freq = int(num_iters * 1. / eval_times)
        for it in range(num_iters):
            
            loss_, _ = self.sess.run([self.losses['mse'], self.train_op], 
                self.get_feed(self.data_gen_train, self.batch_size))
            self.loss_sum += loss_
            if it % self.eval_freq == 0:
                self.do_eval(self.sess, it)
        self.do_eval(self.sess, it)


    def predict(self, user_idxs, movie_idxs):
        n = user_idxs.shape[0]
        print(n)
        num_chunks = int(np.ceil(n * 1. / self.batch_size))
        out = np.array([]).astype(np.float32)
        for chunk in range(num_chunks):
            st = chunk * self.batch_size
            en = min((chunk + 1) * self.batch_size, n)
            out_add = self.sess.run(self.preds, 
                                    {self.user_idxs_in: user_idxs[st:en], 
                                     self.movie_idxs_in: movie_idxs[st:en]})
            
            out = np.append(out, out_add)
        return out

