import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from scipy import sparse
import logging
import pickle
import json
import arviz as az
from lisa import FromRegions, parse_regions_file
from lisa.core.utils import LoadingBar
from collections import Counter
import pickle
from pymc3.variational.callbacks import CheckParametersConvergence
from joblib import Parallel, delayed
import os
import fire

from theano import theano_logger
theano_logger.setLevel(logging.ERROR)

logging.basicConfig()
logger = logging.getLogger('RPModel')
logger.setLevel(logging.INFO)

def clr_transform(x):
        return np.log(x) - np.log(x).mean(axis = -1, keepdims = True)

class TopicRPFeatureFactory:

    rp_range = 40000

    @classmethod
    def fit_models(cls,*, genes, peaks, species, region_topic_distribution, cell_topic_distribution, 
            read_depth, expr_matrix, n_jobs = -1, file_prefix = './', type = 'VI'):

        gene_factory = cls(peaks, species, region_topic_distribution)

        return gene_factory.train_gene_models(genes, cell_topic_distribution=cell_topic_distribution, 
            expr_matrix = expr_matrix, read_depth=read_depth, n_jobs=n_jobs, file_prefix=file_prefix, type=type)

        
    def __init__(self, peaks, species, region_topic_distribution):

        self.regions_test = FromRegions(species, peaks, rp_map='basic', rp_decay=self.rp_range)
        self.rp_map, self.genes, self.peaks = self.regions_test.get_rp_map()

        self.region_score_map = self.regions_test.region_score_map
        self.region_topic_distribution = region_topic_distribution[:, self.region_score_map]


    def get_gene_weights(self, gene_model, cell_topic_distribution):

        return np.dot(cell_topic_distribution, self.region_topic_distribution[:, gene_model.region_mask])


    def get_gene_model(self, gene_symbol, type = 'VI'):

        gene_idx = np.argwhere(np.array([x[3] for x in self.genes]) == gene_symbol)[0]

        region_mask = self.rp_map[gene_idx, :].tocsr().indices

        region_distances = -self.rp_range * np.log2(np.array(self.rp_map[gene_idx, region_mask])).reshape(-1)

        tss_start = self.genes[gene_idx[0]][1]
        strand = self.genes[gene_idx[0]][4]

        upstream_mask = []
        for peak_idx in region_mask:

            peak = self.peaks[peak_idx]
            start, end = peak[1], peak[2]

            if strand == '+':
                upstream_mask.append(start <= tss_start)
            else:
                upstream_mask.append(end >= tss_start)

        promoter_mask = region_distances <= 1500
        upstream_mask = np.logical_and(upstream_mask, ~promoter_mask)
        downstream_mask  = np.logical_and(~upstream_mask, ~promoter_mask)

        region_distances = np.where(np.logical_or(upstream_mask, downstream_mask), region_distances - 1500, region_distances)

        kwargs = dict(
            region_mask = region_mask,
            region_distances = region_distances,
            upstream_mask = upstream_mask,
            promoter_mask = promoter_mask,
            downstream_mask = downstream_mask,
            user_to_model_region_map = self.region_score_map
        )

        if type == 'VI':
            return AssymetricalRPModelVI(gene_symbol, **kwargs)
        elif type == 'TopicMixture':
            return TopicMixtureRPModel(gene_symbol, **kwargs)
        elif type == 'TopicAdjustment':
            return TopicAdjustmentModel(gene_symbol, **kwargs)


    def train_gene_models(self, gene_symbols,*,cell_topic_distribution, expr_matrix, read_depth, n_jobs = -1, 
        file_prefix='./', type = 'TopicMixture', save_model = False):

        if not type == 'TransEffect':

            def train_fn(args):
                model = args[0]
                
                try:
                    model.fit(**args[1], method = 'advi', progressbar=False)
                    model.write_summary(file_prefix + model.gene_symbol + '_expr.json')

                    if save_model:
                        with open(file_prefix + model.gene_symbol + '_model.pkl', 'wb') as f:
                            pickle.dump(model, f)

                    logger.info(model.gene_symbol + ': Done!')
                    return model.gene_symbol + ': Done!'

                except Exception as err:
                    logger.error(model.gene_symbol + ': ' + str(repr(err)))
                    return model.gene_symbol + ': ' + str(repr(err))


            logger.info('Compiling models ...')
            data = []
            for i, symbol in enumerate(gene_symbols):

                try:
                    model = self.get_gene_model(symbol, type = type)
                    weights = self.get_gene_weights(model, cell_topic_distribution)

                    data.append((model, dict(weights = weights, expression = expr_matrix[:,i], read_depth = read_depth, topic_distribution = cell_topic_distribution)))

                except Exception as err:
                    logger.error(symbol + ': ' + str(repr(err)))

            logger.info('Compiled {} models.'.format(len(data)))

            if n_jobs == 1:

                fit_models = []
                for i, gene_data in enumerate(data):
                    fit_models.append(train_fn(gene_data))

                    if i%10 == 0:
                        logger.info('Trained {} models'.format(str(i)))

            else:
                logger.info('Parallelizing training ...')
                fit_models = Parallel(n_jobs= n_jobs, verbose=10)([
                    delayed(train_fn)(gene_data) for gene_data in data
                ])

            return fit_models
    
        else:
            
            model = TransEffectModel().fit(
                weights = None,
                expression = expr_matrix,
                read_depth = read_depth,
                topic_distribution = cell_topic_distribution,
                posterior_samples=100,
            )

            with open(file_prefix + 'trans-effect' + '_model.pkl', 'wb') as f:
                pickle.dump(model, f)

            return ['Done!',]


class RPModelPointEstimator:

    def __init__(self, gene_symbol, *,a_up, a_down, a_promoter, distance_up, distance_down, dropout, theta, b,
            upstream_mask, downstream_mask, promoter_mask, region_distances, region_mask, region_score_map):

        self.a_up = a_up
        self.a_down = a_down
        self.a_promoter = a_promoter
        self.distance_up = distance_up
        self.distance_down = distance_down
        self.dropout = dropout
        self.theta = theta
        self.b = b

        self.upstream_mask = upstream_mask
        self.downstream_mask = downstream_mask
        self.promoter_mask = promoter_mask
        self.region_distances = region_distances
        self.region_mask = region_mask
        self.region_score_map = region_score_map

    def get_log_expr_rate(self, region_weights, return_components = False):

        upstream_weights = region_weights[:,  self.upstream_mask ]
        downstream_weights = region_weights[:, self.downstream_mask ]
        promoter_weights = region_weights[:, self.promoter_mask ]

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        upstream_effects = 1e4 * self.a_up * np.sum(upstream_weights * np.power(0.5, upstream_distances/ (1e3 * np.exp(self.distance_up)) ), axis = 1)
        downstream_effects = 1e4 * self.a_down * np.sum(downstream_weights * np.power(0.5, downstream_distances/ (1e3 * np.exp(self.distance_down))), axis = 1)
        promoter_effects = 1e4 * self.a_promoter * np.sum(promoter_weights, axis = 1)
        
        lam = upstream_effects + promoter_effects + downstream_effects + self.b

        if return_components:
            return lam, upstream_effects, promoter_effects, downstream_effects
        else:
            return lam

    def sample_expression(self, region_weights, cell_read_depth, n_samples = 500):

        expr_rate = np.exp(self.get_log_expr_rate(region_weights))

        return pm.ZeroInflatedNegativeBinomial.dist(mu = cell_read_depth * expr_rate, alpha = self.theta, psi = self.dropout).sample(n_samples)

    def posterior_ISD(self, reg_state, motif_hits):

        assert(motif_hits.shape[1] == reg_state.shape[0])
        assert(len(reg_state.shape) == 1)

        reg_state = reg_state[self.region_score_map][self.region_mask][np.newaxis, :]

        motif_hits = np.array(motif_hits[:,self.region_score_map][:, self.region_mask].todense())

        isd_mask = np.maximum(1 - motif_hits, 0)

        isd_states = np.vstack((reg_state, reg_state * isd_mask))

        rp_scores = np.exp(self.get_log_expr_rate(isd_states))

        return rp_scores[1:], rp_scores[0]

  

class RPModel:

    batch_size = 128
    var_names = []

    def __init__(self, gene_symbol, *, region_mask, region_distances, upstream_mask, downstream_mask, promoter_mask, user_to_model_region_map):

        self.gene_symbol = gene_symbol
        self.region_mask = region_mask
        self.upstream_mask = upstream_mask
        self.downstream_mask = downstream_mask
        self.promoter_mask = promoter_mask
        self.region_distances = region_distances
        self.is_fit = False
        self.region_score_map = user_to_model_region_map

    def summarize_trace(self, model, trace, metric = np.mean):

        summary = dict()
        with model:
            #summary['posterior_summary'] = az.summary(trace, var_names = ['a','b','theta','log_distance']).to_dict()
            for var in  self.var_names:
                summary[var] = metric(trace[var], axis = 0).tolist()
                self.__setattr__(var, np.array(summary[var]))
                summary[var + '_samples'] = trace[var].tolist()

        summary['log_rate_expr_samples'] = self.infer(num_samples=100).tolist()

        return summary

    def write_summary(self, filename):

        with open(filename, 'w') as f:
            json.dump(self.summary, f)

    def load_summary(self, filename):

        with open(filename, 'r') as f:
            self.summary = json.load(f)

        for param in self.var_names:
            self.__setattr__(param, np.array(self.summary[param]))
            self.summary[param + "_samples"] = np.array(self.summary[param + "_samples"])

        self.summary['log_rate_expr_samples'] = np.array(self.summary['log_rate_expr_samples']).mean(axis = 0)

    def fit(self,*, weights, expression, read_depth, topic_distribution, method = 'advi', progressbar=True, posterior_samples = 500, n_steps = 200000):

        expression = np.array(expression).astype(np.float64)
        read_depth = np.array(read_depth).astype(np.int64)
        weights = np.array(weights).astype(np.float64)

        #assert(expression.shape[0] == read_depth.shape[0] == weights.shape[0])
        #assert(len(expression.shape) == 1)
        #assert(len(read_depth.shape) == 1)
        #assert(self.region_distances.shape == (weights.shape[1],))
        #assert(len(weights.shape) == 2)
        #assert(weights.shape[1] == len(self.region_distances))
        
        logging.info('Building model ...')
        self.model = self.make_model(weights = weights, expression = expression,
            read_depth = read_depth, topic_distribution = topic_distribution)
            
        logging.info('Training ...')
        with self.model:
            self.mean_field = pm.fit(n_steps, method=method, progressbar = progressbar,
                callbacks = [CheckParametersConvergence(every=100, tolerance=0.001,diff='relative')])

            self.trace = self.mean_field.sample(posterior_samples)

        self.summary = self.summarize_trace(self.model, self.trace)

        return self

    def infer(self, num_samples = 300):
        return self.sample_node(self.model, self.log_rate_expr, num_samples=num_samples)

    def get_rp_function(self):

        return RPModelPointEstimator(self.gene_symbol,
            a_up = self.a[0],
            a_down = self.a[1],
            a_promoter = self.a[2],
            distance_up = self.log_distance[0],
            distance_down = self.log_distance[1],
            dropout = self.dropout,
            theta = self.theta,
            b = self.b,
            **self.get_rp_function_kwargs()
        )

    def get_rp_function_kwargs(self):
        return dict(
            upstream_mask = self.upstream_mask,
            downstream_mask = self.downstream_mask,
            promoter_mask = self.promoter_mask,
            region_distances = self.region_distances,
            region_mask = self.region_mask,
            region_score_map = self.region_score_map
        )


class TransEffectModel(RPModel):

    var_names = ['beta','b','dropout','theta']
    batch_size = 64

    def __init__(self):
        pass

    def get_rp_function():
        raise NotImplementedError()

    def make_model(self,*, weights, topic_distribution, expression, read_depth):

        K = topic_distribution.shape[-1]
        N,G = expression.shape
        
        self.mixing_weights = pm.Minibatch(topic_distribution, batch_size=self.batch_size)
        self.read_depth_batches = pm.Minibatch(read_depth[:,np.newaxis].copy(), batch_size=self.batch_size)
        self.expression_batches = pm.Minibatch(expression, batch_size=self.batch_size)
        
        with pm.Model() as model:

            beta = pm.Gamma('beta', alpha = 1, beta = 1, shape = (K,G))

            b = pm.Normal('b', mu = 0, sigma = 100, shape = (1,G))

            self.log_rate_expr = tt.dot(self.mixing_weights, beta) + b # (N,K) * (K,G) + (1,G) -> (N,G)

            dropout = pm.Beta('dropout', alpha = 1, beta = 10, testval = 0.2, shape = (1,G))

            dispersion = pm.Gamma('theta', alpha = 2, beta = 2, testval = 2, shape = (1,G))
            
            X = pm.ZeroInflatedNegativeBinomial('expr', mu = tt.addbroadcast(self.read_depth_batches, 1) * tt.exp(self.log_rate_expr), psi = dropout, 
                            alpha = dispersion, observed = self.expression_batches, total_size = N)
            
        return model

    def sample_node(self, model, node, num_samples = 300):
        with model:
            return self.mean_field.sample_node(node, 
                    more_replacements={
                        self.read_depth_batches.minibatch : self.read_depth_batches.shared,
                        self.mixing_weights.minibatch : self.mixing_weights.shared,
                        self.expression_batches.minibatch : self.expression_batches.shared
                    }, size = num_samples).eval()

    def get_log_expr_rate(self, topic_distribution, region_weights):

        return np.dot(topic_distribution, self.beta) + self.b


class TopicMixtureRPModel(RPModel):

    batch_size = 128
    var_names = ['a','b','theta','log_distance','a_hyper','tau','dropout']

    def make_model(self,*, weights, topic_distribution, expression, read_depth):

        self.K = topic_distribution.shape[1]
        N = weights.shape[0]

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        self.upstream_batches, self.downstream_batches, self.promoter_batches, self.expression_batches, self.read_depth_batches = \
            pm.Minibatch(weights[:,  self.upstream_mask ].copy(), self.batch_size), pm.Minibatch(weights[:, self.downstream_mask ].copy(), self.batch_size), \
            pm.Minibatch(weights[:, self.promoter_mask ].copy(), self.batch_size), pm.Minibatch(expression, self.batch_size), pm.Minibatch(read_depth, self.batch_size)

        self.mixing_weights = pm.Minibatch(topic_distribution, self.batch_size)

        #def RP(reg_state, region_distance, decay):
        #    N, D = reg_state.shape
        #    return tt.sum(reg_state.reshape((N,D,1)) * tt.power(0.5, region_distance.reshape((1,-1, 1)) / (1e3 * tt.exp( decay.reshape((1,1,-1))))), axis = 1)

        def RP(w, d, l):
            return tt.sum(w * tt.power(0.5, d.reshape((1,-1)) / (1e3 * tt.exp(l)) ), axis = 1).reshape((-1,1))

        with pm.Model() as model:
            
            d = pm.Normal('log_distance', mu = np.e, sigma = 2, shape = 2, testval=np.e)

            a_hyper = pm.HalfNormal('a_hyper', sigma = 12, shape = (3,1))
            tau = pm.Gamma('tau', alpha = 1, beta = 5, shape = (3,1))
            a = pm.TruncatedNormal('a', sigma = tau, mu = a_hyper, shape = (3,self.K), lower = 0)
            
            b = pm.Normal('b', sigma = 15, mu = 0, shape = 1, testval=-10)

            theta = pm.Gamma('theta', alpha = 2, beta = 1/2, shape = 1, testval = 2)

            rp_upstream = 1e4 * a[0,:] * RP(self.upstream_batches, upstream_distances, d[0]) # n,K
            rp_downstream = 1e4 * a[1,:] * RP(self.downstream_batches, downstream_distances, d[1])
            rp_promoter = 1e4 * a[2,:] * tt.sum(self.promoter_batches, axis = 1).reshape((-1,1))
            
            self.log_rate_expr = tt.sum((rp_upstream + rp_promoter + rp_downstream) * self.mixing_weights, axis = 1) + b
            
            dropout = pm.Beta('dropout', alpha = 1, beta = 10)
                
            X = pm.ZeroInflatedNegativeBinomial('expr', mu = self.read_depth_batches * tt.exp(self.log_rate_expr), psi = dropout,
                            alpha = theta, observed = self.expression_batches, total_size = N)

        return model

    def sample_node(self, model, node, num_samples = 300):
        with model:
            return self.mean_field.sample_node(node, 
                    more_replacements={
                        self.upstream_batches.minibatch : self.upstream_batches.shared,
                        self.downstream_batches.minibatch : self.downstream_batches.shared,
                        self.promoter_batches.minibatch : self.promoter_batches.shared,
                        self.read_depth_batches.minibatch : self.read_depth_batches.shared,
                        self.mixing_weights.minibatch : self.mixing_weights.shared,
                        self.expression_batches.minibatch : self.expression_batches.shared
                    }, size = num_samples).eval()

    def get_log_expr_rate(self, topic_distribution, region_weights):

        unmixed_log_rate = np.hstack([
            self.get_topic_level_function(i, b = 0).get_log_expr_rate(region_weights)[:,np.newaxis] for i in range(topic_distribution.shape[-1])
        ])

        return (unmixed_log_rate * topic_distribution).sum(axis = 1) + self.b


    def sample_rp_from_hyperprior(self, region_weights):

        estimated_epxr = []
        for i in range(self.summary['a_hyper_samples'].shape[0]):

            a_samples = self.summary['a_hyper_samples'][i,:,0]
            estimated_epxr.append(
                RPModelPointEstimator(self.gene_symbol,
                    a_up = a_samples[0],
                    a_down = a_samples[1],
                    a_promoter = a_samples[2],
                    distance_up = self.log_distance[0],
                    distance_down = self.log_distance[1],
                    dropout = self.dropout,
                    theta = self.theta,
                    b = self.b,
                    **self.get_rp_function_kwargs()
                ).get_log_expr_rate(region_weights)[:,np.newaxis]
            )

        return np.hstack(estimated_epxr).mean(axis = 1)


    def get_hyperprior_function(self):

        return RPModelPointEstimator(self.gene_symbol,
            a_up = self.a_hyper[0][0],
            a_down = self.a_hyper[1][0],
            a_promoter = self.a_hyper[2][0],
            distance_up = self.log_distance[0],
            distance_down = self.log_distance[1],
            dropout = self.dropout,
            theta = self.theta,
            b = self.b,
            **self.get_rp_function_kwargs()
        )

    def get_topic_level_function(self, topic_num, b = None):

        return RPModelPointEstimator(self.gene_symbol,
            a_up = self.a[0][topic_num],
            a_down = self.a[1][topic_num],
            a_promoter = self.a[2][topic_num],
            distance_up = self.log_distance[0],
            distance_down = self.log_distance[1],
            dropout = self.dropout,
            theta = self.theta,
            b = self.b if b is None else b,
            **self.get_rp_function_kwargs(),
        )


class AssymetricalRPModelVI(RPModel):

    batch_size = 256
    var_names = ['a','b','theta','log_distance','dropout']

    #____ MODEL TRAINING _____
    def make_model(self,*, weights, topic_distribution, expression, read_depth):

        def RP(w, d, l):
            return tt.sum(w * tt.power(0.5, d.reshape((1,-1)) / (1e3 * tt.exp(l)) ), axis = 1).reshape((-1,))

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        self.upstream_batches, self.downstream_batches, self.promoter_batches, self.expression_batches, self.read_depth_batches = \
            pm.Minibatch(weights[:,  self.upstream_mask ].copy(), self.batch_size), pm.Minibatch(weights[:, self.downstream_mask ].copy(), self.batch_size), \
            pm.Minibatch(weights[:, self.promoter_mask ].copy(), self.batch_size), pm.Minibatch(expression, self.batch_size), pm.Minibatch(read_depth, self.batch_size)

        with pm.Model() as model:
            
            d = pm.Normal('log_distance', mu = np.e, sigma = 2, shape = 2, testval=np.e)
            
            pm.Deterministic('distance', tt.exp(d))
            
            a = pm.HalfNormal('a', sigma=12, shape = 3, testval = 10)
            b = pm.Normal('b', sigma = 15, mu = 0, shape = 1, testval=-10)
            
            theta = pm.Gamma('theta', alpha = 2, beta = 1/2, shape = 1, testval = 2)
            
            rp_upstream = 1e4 * a[0] * RP(self.upstream_batches, upstream_distances, d[0])
            rp_downstream = 1e4 * a[1] * RP(self.downstream_batches, downstream_distances, d[1])
            rp_promoter = 1e4 * a[2] * tt.sum(self.promoter_batches, axis = 1).reshape((-1,))
            
            self.log_rate_expr = rp_upstream + rp_promoter + rp_downstream + b
            
            dropout = pm.Beta('dropout', alpha = 1, beta = 10)
                
            X = pm.ZeroInflatedNegativeBinomial('expr', mu = self.read_depth_batches * tt.exp(self.log_rate_expr), psi = dropout,
                            alpha = theta, observed = self.expression_batches, total_size = len(expression))

        return model

    def sample_node(self, model, node, num_samples = 300):
        with model:
            return self.mean_field.sample_node(node, 
                    more_replacements={
                        self.upstream_batches.minibatch : self.upstream_batches.shared,
                        self.downstream_batches.minibatch : self.downstream_batches.shared,
                        self.promoter_batches.minibatch : self.promoter_batches.shared,
                        self.read_depth_batches.minibatch : self.read_depth_batches.shared,
                        self.expression_batches.minibatch : self.expression_batches.shared
                    }, size = num_samples).eval()

    def get_log_expr_rate(self, topic_distribution, region_weights):
        return self.get_rp_function().get_log_expr_rate(region_weights)


class TopicAdjustmentModel(RPModel):

    batch_size = 256
    var_names = ['a','b','theta','log_distance','dropout','beta']

    #____ MODEL TRAINING _____
    def make_model(self,*, weights, topic_distribution, expression, read_depth):

        def RP(w, d, l):
            return tt.sum(w * tt.power(0.5, d.reshape((1,-1)) / (1e3 * tt.exp(l)) ), axis = 1).reshape((-1,))

        K = topic_distribution.shape[1]

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        self.upstream_batches, self.downstream_batches, self.promoter_batches, self.expression_batches, self.read_depth_batches = \
            pm.Minibatch(weights[:,  self.upstream_mask ].copy(), self.batch_size), pm.Minibatch(weights[:, self.downstream_mask ].copy(), self.batch_size), \
            pm.Minibatch(weights[:, self.promoter_mask ].copy(), self.batch_size), pm.Minibatch(expression, self.batch_size), pm.Minibatch(read_depth, self.batch_size)

        self.topic_weight_batches = pm.Minibatch(clr_transform(topic_distribution).copy(), self.batch_size)

        with pm.Model() as model:
            
            d = pm.Normal('log_distance', mu = np.e, sigma = 2, shape = 2, testval=np.e)
            
            pm.Deterministic('distance', tt.exp(d))
            
            a = pm.HalfNormal('a', sigma=12, shape = 3, testval = 10)
            #a = pm.Gamma('a', alpha = 1, beta = 1/10, testval = 10, shape = 3)
            b = pm.Normal('b', sigma = 15, mu = 0, shape = 1, testval=-10)
            
            theta = pm.Gamma('theta', alpha = 2, beta = 1/2, shape = 1, testval = 2)
            
            rp_upstream = 1e4 * a[0] * RP(self.upstream_batches, upstream_distances, d[0])
            rp_downstream = 1e4 * a[1] * RP(self.downstream_batches, downstream_distances, d[1])
            rp_promoter = 1e4 * a[2] * tt.sum(self.promoter_batches, axis = 1).reshape((-1,))

            beta = pm.Laplace('beta', mu = 0, b = 1/10, shape = (K,1), testval=0.01)

            self.log_rate_expr = rp_upstream + rp_promoter + rp_downstream + tt.reshape(tt.dot(self.topic_weight_batches, beta), (-1,)) + b

            dropout = pm.Beta('dropout', alpha = 1, beta = 10)
                
            X = pm.ZeroInflatedNegativeBinomial('expr', mu = self.read_depth_batches * tt.exp(self.log_rate_expr), psi = dropout,
                            alpha = theta, observed = self.expression_batches, total_size = len(expression))

        return model

    def sample_node(self, model, node, num_samples = 300):
        with model:
            return self.mean_field.sample_node(node, 
                    more_replacements={
                        self.upstream_batches.minibatch : self.upstream_batches.shared,
                        self.downstream_batches.minibatch : self.downstream_batches.shared,
                        self.promoter_batches.minibatch : self.promoter_batches.shared,
                        self.read_depth_batches.minibatch : self.read_depth_batches.shared,
                        self.expression_batches.minibatch : self.expression_batches.shared,
                        self.topic_weight_batches.minibatch : self.topic_weight_batches.shared,
                    }, size = num_samples).eval()

    def get_log_expr_rate(self, topic_distribution, region_weights):
        return self.get_rp_function().get_log_expr_rate(region_weights) + np.dot(clr_transform(topic_distribution), self.beta).reshape(-1)


'''class ConvergenceError(Exception):
    pass

class AssymetricalRPModel:

    def __init__(self, gene_symbol, *, region_mask, region_distances, upstream_mask, downstream_mask, promoter_mask, user_to_model_region_map):

        self.gene_symbol = gene_symbol
        self.region_mask = region_mask
        self.upstream_mask = upstream_mask
        self.downstream_mask = downstream_mask
        self.promoter_mask = promoter_mask
        self.region_distances = region_distances
        self.is_fit = False
        self.region_score_map = user_to_model_region_map

    #____ MODEL TRAINING _____
    def make_model(self, weights, expression, read_depth):

        threshold = np.log((expression/read_depth).max() * 1.5)

        def RP(w, d, phi):
            return tt.sum(w * tt.power(phi, d.reshape((1,-1))/1e4), axis = 1).reshape((-1,))

        upstream_weights = weights[:,  self.upstream_mask ]
        downstream_weights = weights[:, self.downstream_mask ]
        promoter_weights = weights[:, self.promoter_mask ]

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        with pm.Model() as model:
            
            phi_j = pm.Beta("phi", alpha=5, beta=5, shape=2, testval = 0.5)

            pm.Deterministic('distance', 10 * tt.log(0.5)/tt.log(phi_j))
            
            # RP link hyperpriors
            a_j = pm.HalfNormal('a', sigma=40000, shape = 3, testval=10000)
            #a_j = pm.Gamma('a', alpha = 1.1, beta = 30000, shape = 3, testval=10000)
            b_j = pm.Normal('b', mu = 0, sigma = 15, testval = 0, shape = 1)
            
            # Calculate RP
            rp_upstream = a_j[0] * RP(upstream_weights, upstream_distances, phi_j[0])
            rp_downstream = a_j[1] * RP(downstream_weights, downstream_distances, phi_j[1])
            rp_promoter = a_j[2] * tt.sum(promoter_weights, axis = 1).reshape((-1,))
            
            log_rate_expr = rp_upstream + rp_promoter + rp_downstream + b_j

            pm.Potential('lambda_constraint', pm.Gamma.dist(alpha=1, beta=1).logp(tt.maximum(log_rate_expr - threshold, 0)))
            #tt.printing.Print('pot')(log_rate_expr - threshold)
            
            #theta_j = pm.HalfNormal("theta", sigma = 5, shape = 1)
            theta_j = pm.Gamma("theta", alpha = 2, beta = 2, shape = 1)
            # Sample observed gene expression
            X_ij = pm.NegativeBinomial('expr', mu = read_depth * tt.exp(log_rate_expr), alpha = theta_j, observed = expression)
            
        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        if 'model' in state:
            del state['model']
        if 'trace' in state:
            del state['trace']
            
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    def persist(self, filename, save_trace = False):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    def compute_lambda_mean_posterior(self, region_weights, return_components = False):

        upstream_weights = region_weights[:,  self.upstream_mask ]
        downstream_weights = region_weights[:, self.downstream_mask ]
        promoter_weights = region_weights[:, self.promoter_mask ]

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        a, phi, b = self.a, self.phi, self.b

        upstream_effects = a[0] * np.sum(upstream_weights * np.power(phi[0], upstream_distances/1e4), axis = 1)
        downstream_effects = a[1] * np.sum(downstream_weights * np.power(phi[1], downstream_distances/1e4), axis = 1)
        promoter_effects = a[2] * np.sum(promoter_weights, axis = 1)
        
        lam = np.exp(upstream_effects + promoter_effects + downstream_effects + b[0])

        if return_components:
            return lam, upstream_effects, promoter_effects, downstream_effects
        else:
            return lam
            

    def estimate_logp(self, lambd, cell_read_depth, num_transcripts):

        return pm.NegativeBinomial.dist(mu = cell_read_depth * lambd, alpha = self.theta[0]).logp(num_transcripts).eval()

    #___ POSTERIOR SAMPLING ___

    def sample_posterior(self, lambd, cell_read_depth, draws_per_sample = 1):

        draws = pm.NegativeBinomial.dist(mu = cell_read_depth * lambd, alpha = self.theta[0]).random(size = draws_per_sample)

        return draws
    
class AssymetricalRPModelMAP(AssymetricalRPModel):


    def fit(self, weights, expression, read_depth, tries = 10):

        expression = np.array(expression).astype(np.float64)
        read_depth = np.array(read_depth).astype(np.int64)
        weights = np.array(weights).astype(np.float64)

        assert(expression.shape[0] == read_depth.shape[0] == weights.shape[0])
        assert(len(expression.shape) == 1)
        assert(len(read_depth.shape) == 1)
        assert(self.region_distances.shape == (weights.shape[1],))
        assert(len(weights.shape) == 2)
        assert(weights.shape[1] == len(self.region_distances))

        converged, attempts = False, 0

        while not converged and attempts < tries:

            shuffle = np.random.choice(len(expression), size = len(expression))
        
            logging.info('Building model ...')
            self.model = self.make_model(weights[shuffle], expression[shuffle], read_depth[shuffle])
            
            logging.info('Training ...')
            with self.model:

                attempts+=1

                self.map = pm.find_MAP(progressbar=False, model = self.model)

                if (self.map['a'] < 1e6).all() and self.map['theta'] < 1e4:
                    converged=True
                elif attempts < tries:
                    logging.warn('{} MAP estimate did not converge. Trying attempt {}/{}'.format(self.gene_symbol, str(attempts+1), str(tries)))

        if not converged:
            logging.error('{} MAP estimate did not converge after {} tries.'.format(self.gene_symbol, str(attempts)))
            raise ConvergenceError('MAP estimate Failed to converge.') 
        
        self.set_map(self.map)

        return self

    def set_map(self, map_estimate):
        for param, vals in map_estimate.items():
            self.__setattr__(param, np.array(vals))
        
        self.is_fit = True

    def load_summary(self, filename):

        with open(filename, 'r') as f:
            self.map = json.load(f)

        self.set_map(self.map)

    def write_summary(self, filename):
    
        with open(filename, 'w') as f:
            write_dict = dict()
            for param, vals in self.map.items():
                write_dict[param] = vals.tolist()
                
            json.dump(write_dict, f, indent=4)


class AssymetricalRPModelMCMC(AssymetricalRPModel):

    
    #___ SUMMARY PERSISTENCE ____
    def add_posterior_predictive_distribution(self, model_summary):
        
        posterior_samples = model_summary['posterior_samples']
        self.phi = np.array(posterior_samples['phi'])
        self.distance = np.array(posterior_samples['distance'])
        self.a = np.array(posterior_samples['a'])
        self.b = np.array(posterior_samples['b'])
        self.theta = np.array(posterior_samples['theta'])
        self.is_fit = True

    def fit(self, weights, expression, read_depth):

        expression = np.array(expression).astype(np.float64)
        read_depth = np.array(read_depth).astype(np.int64)
        weights = np.array(weights).astype(np.float64)

        assert(expression.shape[0] == read_depth.shape[0] == weights.shape[0])
        assert(len(expression.shape) == 1)
        assert(len(read_depth.shape) == 1)
        assert(self.region_distances.shape == (weights.shape[1],))
        assert(len(weights.shape) == 2)
        assert(weights.shape[1] == len(self.region_distances))

        logging.info('Building model ...')
        self.model = self.make_model(weights, expression, read_depth)
        
        logging.info('Training ...')
        with self.model:

            self.trace = pm.sample(tune=3000, draws = 1000, chains = 2)

            logging.info('Sampling posterior predictive distribution ...')
        
            logging.info('Sampling posterior predictive distribution ...')
            for param, draws in pm.sample_posterior_predictive(self.trace, var_names=['phi','distance','theta','a','b']).items():
                self.__setattr__(param, np.random.choice(draws.reshape(-1), 1000))            

        return self

    def load_posterior_predictive_distribution(self, model_summary_file):
        with open(model_summary_file, 'r') as f:
            self.add_posterior_predictive_distribution( json.load(f) )


    @staticmethod
    def _RP(weights, distances, phi):
        return np.sum(weights.reshape(1,-1) * np.power(phi.reshape(-1,1), 
            distances.reshape(1,-1)/1e4), axis = 1).reshape(-1)


    def compute_lambda_mean_posterior(self, region_weights, return_components = False):

        phi = self.phi.mean(axis = 0)
        a = self.a.mean(axis = 0)
        b = self.b.mean()

        upstream_weights = region_weights[:,  self.upstream_mask ]
        downstream_weights = region_weights[:, self.downstream_mask ]
        promoter_weights = region_weights[:, self.promoter_mask ]

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        upstream_effects = a[0] * np.sum(upstream_weights * np.power(phi[0], upstream_distances/1e4), axis = 1)
        promoter_effects = a[2] * np.sum(promoter_weights, axis = 1)
        downstream_effects = a[1] * np.sum(downstream_weights * np.power(phi[1], downstream_distances/1e4), axis = 1)
        
        lam = np.exp(upstream_effects + promoter_effects + downstream_effects + b)

        if return_components:
            return lam, upstream_effects, promoter_effects, downstream_effects
        else:
            return lam

    def estimate_logp(self, lambd, cell_read_depth, num_transcripts):

        return pm.NegativeBinomial.dist(mu = cell_read_depth * lambd, alpha = self.theta.mean()).logp(num_transcripts).eval()


    def is_not_bimodal(self):
        
        for phi in self.phi.T:
            a,b,_,_ = beta.fit(phi, floc= 0, fscale = 1)
            if a < 1 and b < 1:
                return False

        return True

    #___ POSTERIOR SAMPLING ___

    def sample_posterior(self, lambd, cell_read_depth, draws_per_sample = 1):

        draws = pm.NegativeBinomial.dist(mu = cell_read_depth * lambd, alpha = self.theta.mean()).random(size = draws_per_sample)

        return draws

    #___ DISTRIBUTION COMPARISON ___

    @staticmethod
    def hellinger_dist(p,q):
        assert(len(p) == len(q))
        return 1/np.sqrt(2) * np.sqrt( np.square(np.sqrt(q) - np.sqrt(p)).sum() )


    def compute_posterior_sample_hellinger_distance(self, reference_distribution, compare_distribution):

        p_counts = Counter(reference_distribution)
        q_counts = Counter(compare_distribution)

        all_discrete_values = set(list(p_counts.keys()) + list(q_counts.keys()))

        p, q = [],[]
        for val in all_discrete_values:
            p.append(p_counts[val]/len(reference_distribution))
            q.append(q_counts[val]/len(compare_distribution))
            
        p,q = np.array(p), np.array(q)

        return self.hellinger_dist(p,q)


    def get_phi_histogram(self, num_bins = 50):

        vals, bins = np.histogram(self.phi, bins = np.linspace(0,1,num_bins+1))

        return vals/len(self.phi)'''


def main(*,genes_file, species, peaks_file, read_depth_array, expr_matrix, cell_topic_distribution_matrix, 
    region_topic_distribution_matrix, n_jobs = -1, file_prefix='./', start_idx = 0, end_idx = None, type = 'VI'):

    peaks, _ = parse_regions_file(peaks_file)

    cell_topic_distribution = np.load(cell_topic_distribution_matrix)
    region_topic_distribution = np.load(region_topic_distribution_matrix)
    read_depth = np.load(read_depth_array)
    expr = np.load(expr_matrix)

    with open(genes_file, 'r') as f:
        genes = [x.strip().upper() for x in f.readlines()]

    results = TopicRPFeatureFactory.fit_models(
        genes = genes[int(start_idx) : int(end_idx) if not end_idx is None else None],
        peaks = peaks,
        species = species,
        read_depth = read_depth,
        region_topic_distribution = region_topic_distribution,
        cell_topic_distribution = cell_topic_distribution,
        expr_matrix = expr[:,start_idx:end_idx],
        n_jobs= int(n_jobs),
        file_prefix= file_prefix,
        type = type,
    )

    print('\n'.join(map(str, results)))

if __name__ == "__main__":
    fire.Fire(main)