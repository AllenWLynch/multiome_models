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


class TopicRPFeatureFactory:

    rp_range = 40000

    @classmethod
    def fit_models(cls,*, genes, peaks, species, region_topic_distribution, cell_topic_distribution, 
            read_depth, expr_matrix, n_jobs = -1, file_prefix = './'):

        gene_factory = cls(peaks, species, region_topic_distribution)

        return gene_factory.train_gene_models(genes, cell_topic_distribution=cell_topic_distribution, 
            expr_matrix = expr_matrix, read_depth=read_depth, n_jobs=n_jobs, file_prefix=file_prefix)

        
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

        if type == 'MAP':
            return AssymetricalRPModelMAP(gene_symbol, **kwargs)
        elif type == 'MCMC':
            return AssymetricalRPModelMCMC(gene_symbol, **kwargs)
        else:
            return AssymetricalRPModelVI(gene_symbol, **kwargs)

    def train_gene_models(self, gene_symbols,*,cell_topic_distribution, expr_matrix, read_depth, n_jobs = -1, file_prefix='./'):

        def train_fn(args):

            try:
            
                model, weights, expr, read_depth = args

                model.fit(weights, expr, read_depth)

                model.write_summary(file_prefix + model.gene_symbol + '_expr.json')

                logger.info(model.gene_symbol + ': Done!')

            except Exception as err:
                logger.error(model.gene_symbol + ': ' + str(repr(err)))


        logger.info('Compiling models ...')
        data = []
        for i, symbol in enumerate(gene_symbols):

            try:
                model = self.get_gene_model(symbol)
                weights = self.get_gene_weights(model, cell_topic_distribution)

                data.append((model, weights, expr_matrix[:,i], read_depth))
            except Exception as err:
                logger.error(symbol + ': ' + str(repr(err)))

        logger.info('Compiled {} models.'.format(len(data)))
        logger.info('Parallelizing training ...')
        fit_models = Parallel(n_jobs= n_jobs, verbose=10)([
            delayed(train_fn)(gene_data) for gene_data in data
        ])

        return fit_models

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


    def posterior_ISD(self, reg_state, motif_hits):

        assert(motif_hits.shape[1] == reg_state.shape[0])
        assert(len(reg_state.shape) == 1)

        reg_state = reg_state[self.region_score_map][self.region_mask][np.newaxis, :]

        motif_hits = np.array(motif_hits[:,self.region_score_map][:, self.region_mask].todense())

        isd_mask = np.maximum(1 - motif_hits, 0)

        isd_states = np.vstack((reg_state, reg_state * isd_mask))

        _, upstream, promoter, downstream = self.compute_lambda_mean_posterior(isd_states, return_components=True)

        rp_scores = upstream + promoter + downstream

        return np.sqrt(1 - rp_scores[1:]/rp_scores[0])

class ConvergenceError(Exception):
    pass

class AssymetricalRPModelVI(AssymetricalRPModel):

    batch_size = 256
    var_names = ['a','b','theta','log_distance']

    #____ MODEL TRAINING _____
    def make_model(self, weights, expression, read_depth):

        def RP(w, d, l):
            return tt.sum(w * tt.power(0.5, d.reshape((1,-1)) / (1e3 * tt.exp(l)) ), axis = 1).reshape((-1,))

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        upstream_batches, downstream_batches, promoter_batches, expression_batches, read_depth_batches = \
            pm.Minibatch(weights[:,  self.upstream_mask ].copy(), self.batch_size), pm.Minibatch(weights[:, self.downstream_mask ].copy(), self.batch_size), \
            pm.Minibatch(weights[:, self.promoter_mask ].copy(), self.batch_size), pm.Minibatch(expression, self.batch_size), pm.Minibatch(read_depth, self.batch_size)

        logger.debug('Upstream weights shape: ' + str(weights[:,  self.upstream_mask ].copy().shape))
        logger.debug('Downstream weights shape: ' + str(weights[:,  self.downstream_mask ].copy().shape))
        logger.debug('Upstream weights shape: ' + str(weights[:,  self.promoter_mask].copy().shape))
        logger.debug('Batch size: ' + str(self.batch_size))

        with pm.Model() as model:
            
            d = pm.Normal('log_distance', mu = np.e, sigma = 1, shape = 2, testval=np.e)
            
            pm.Deterministic('distance', tt.exp(d))
            
            a = pm.HalfNormal('a', sigma=10000, shape = 3, testval = 1e3)
            b = pm.Normal('b', sigma = 15, mu = 0, shape = 1, testval=-10)
            
            theta = pm.Gamma('theta', alpha = 2, beta = 2, shape = 1, testval = 2)
            
            rp_upstream = a[0] * RP(upstream_batches, upstream_distances, d[0])
            rp_downstream = a[1] * RP(downstream_batches, downstream_distances, d[1])
            rp_promoter = a[2] * tt.sum(promoter_batches, axis = 1).reshape((-1,))
            
            log_rate_expr = rp_upstream + rp_promoter + rp_downstream + b
            
            # Sample observed gene expression
            X = pm.NegativeBinomial('expr', mu = read_depth_batches * tt.exp(log_rate_expr), alpha = theta, 
                observed = expression_batches, total_size = len(expression))

        return model


    def summarize_trace(self, model, trace, metric = np.mean):

        summary = dict()
        with model:
            #summary['posterior_summary'] = az.summary(trace, var_names = ['a','b','theta','log_distance']).to_dict()
            for var in  self.var_names:
                summary[var] = metric(trace[var], axis = 0).tolist()

                summary[var + '_samples'] = trace[var].tolist()

        return summary

    def write_summary(self, filename):

        with open(filename, 'w') as f:
            json.dump(self.summary, f)

    def load_summary(self, filename):

        with open(filename, 'r') as f:
            self.summary = json.load(f)

        for param in self.var_names:
            self.__setattr__(param, self.summary[param])


    def fit(self, weights, expression, read_depth, method = 'advi'):

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
            mean_field = pm.fit(200000, method=method, progressbar = False,
                callbacks = [CheckParametersConvergence(every=100, tolerance=0.001,diff='relative')])

            self.trace = mean_field.sample(500)

        self.summary = self.summarize_trace(self.model, self.trace)

        for param in self.var_names:
            self.__setattr__(param, self.summary[param])

        return self

    def compute_lambda_mean_posterior(self, region_weights, return_components = False):

        upstream_weights = region_weights[:,  self.upstream_mask ]
        downstream_weights = region_weights[:, self.downstream_mask ]
        promoter_weights = region_weights[:, self.promoter_mask ]

        upstream_distances = self.region_distances[self.upstream_mask]
        downstream_distances = self.region_distances[self.downstream_mask]

        a, d, b = self.a, self.log_distance, self.b

        upstream_effects = a[0] * np.sum(upstream_weights * np.power(0.5, upstream_distances/ (1e3 * np.exp(d[0])) ), axis = 1)
        downstream_effects = a[1] * np.sum(downstream_weights * np.power(0.5, downstream_distances/ (1e3 * np.exp(d[1]))), axis = 1)
        promoter_effects = a[2] * np.sum(promoter_weights, axis = 1)
        
        lam = np.exp(upstream_effects + promoter_effects + downstream_effects + b[0])

        if return_components:
            return lam, upstream_effects, promoter_effects, downstream_effects
        else:
            return lam
        
    
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

        return vals/len(self.phi)


def main(*,genes_file, species, peaks_file, read_depth_array, expr_matrix, cell_topic_distribution_matrix, 
    region_topic_distribution_matrix, n_jobs = -1, file_prefix='./'):

    peaks, _ = parse_regions_file(peaks_file)

    cell_topic_distribution = np.load(cell_topic_distribution_matrix)
    region_topic_distribution = np.load(region_topic_distribution_matrix)
    read_depth = np.load(read_depth_array)
    expr = np.load(expr_matrix)

    with open(genes_file, 'r') as f:
        genes = [x.strip().upper() for x in f.readlines()]

    TopicRPFeatureFactory.fit_models(
        genes = genes,
        peaks = peaks,
        species = species,
        read_depth = read_depth,
        region_topic_distribution = region_topic_distribution,
        cell_topic_distribution = cell_topic_distribution,
        expr_matrix = expr,
        n_jobs= n_jobs,
        file_prefix= file_prefix
    )

if __name__ == "__main__":
    fire.Fire(main)