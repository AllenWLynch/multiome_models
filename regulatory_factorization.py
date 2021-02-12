import pymc3 as pm
import numpy as np
import logging
import theano
import time

theano.config.compute_test_value = 'ignore'
logger = logging.getLogger()
logger.setLevel(logging.INFO)

### How to compute gamma priors?
class RegulatoryFactorizer:

    def __init__(self, reg_topics, expr_data, include_intercept = True, topic_rp_distribution = None, std_mult = 5, read_depth = None):
        
        expr_data = np.array(expr_data)
        reg_topics = np.array(reg_topics)
        if not topic_rp_distribution is None:
            topic_rp_distribution = np.array(topic_rp_distribution)

        assert(reg_topics.shape[0] > reg_topics.shape[1]), 'Topics must be axis 1 of array, cells axis 0'
        assert(reg_topics.shape[0] == expr_data.shape[0])
        if not topic_rp_distribution is None:
            assert(topic_rp_distribution.shape[0] == reg_topics.shape[1]), 'Topic_rp_distribution (topics x genes) must have same number of topics as reg_topic matrix (cells x topics).'

        self.include_intercept = include_intercept
        self.num_cells, self.num_topics = reg_topics.shape

        #sum across expr data to get read depth of cells
        if read_depth is None:
            self.read_depth = expr_data.sum(axis = 1)[:, np.newaxis]
        else:
            assert(len(read_depth.shape) == 1)
            assert(len(read_depth) == self.num_cells)
            self.read_depth = read_depth[:, np.newaxis]

        self.lambda_prior, self.lambda_null = self.make_lambda_priors(reg_topics, expr_data, topic_rp_distribution, self.read_depth) #genes x 

        if self.include_intercept:
            #add intercept column of all ones to the left factorized values
            reg_topics = np.hstack([reg_topics, np.ones(self.num_cells)[:, np.newaxis]])
        
        self.reg_topics = reg_topics
        self.expr_data = expr_data
        #self.std_prior = np.repeat(lambda_null*5, reg_topics.shape[1] + 1 if self.include_intercept else 0, axis = 0)
        self.std_mult = std_mult

        self.model = self.build_model(read_depth = self.read_depth, reg_topics = self.reg_topics, expr_data = self.expr_data, lambda_prior = self.lambda_prior, std_mult = self.std_mult)


    def make_lambda_priors(self, reg_topics, expr_data, topic_rp_distribution, read_depth):

        if topic_rp_distribution is None:
            rp_weights = np.ones((expr_data.shape[1], self.num_topics + (1 if self.include_intercept else 0))).T #topics x genes

        else:
            if not np.isclose(topic_rp_distribution.sum(axis = 0), 1).all():
                topic_rp_distribution = topic_rp_distribution/topic_rp_distribution.sum(axis = 0, keepdims = True)

            rp_weights = np.multiply(topic_rp_distribution, self.num_topics) # topics x genes

            if self.include_intercept:
                rp_weights = np.vstack([rp_weights, np.ones((1, rp_weights.shape[1]))])

        self.rp_weights = rp_weights

        lambda_null = expr_data.sum(axis = 0, keepdims = True)/read_depth.sum() # 1 x genes

        lambda_adj = np.multiply(rp_weights, lambda_null) # topics x genes

        return lambda_adj, lambda_null

    @staticmethod
    def build_model(*,read_depth, reg_topics, expr_data, lambda_prior, std_mult):

        #expr_factor_shape = (reg_topics.shape[1], expr_data.shape[1])
        
        beta = 1/std_mult
        alpha = lambda_prior * beta

        logging.info('Building model')
        with pm.Model() as model:

            expr_topics = pm.Gamma('gamma', alpha = alpha, beta = beta, shape = lambda_prior.shape)

            response = pm.Poisson('response',
                read_depth * pm.math.dot(reg_topics, expr_topics),
                observed = expr_data
            )
        logging.info('Done building model')
        return model
    

    def find_map(self):
        tstart = time.time()
        with self.model:
            logging.info('finding PMF MAP using L-BFGS-B optimization...')
            self._map = pm.find_MAP(method='L-BFGS-B')

        elapsed = int(time.time() - tstart)
        logging.info('found PMF MAP in {} seconds'.format(str(elapsed)))
        return self._map

    def get_data_log_likelihood(self):

        try:
            self._map
        except AttributeError:
            raise AssertionError('MAP must be computed before estimating likelihood')
            
        map_poisson_params = np.multiply(self.read_depth, np.dot(self.reg_topics, self._map['gamma']))

        return pm.Poisson(map_poisson_params).logp(self.expr_data)