import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from scipy import sparse
import logging
import pickle
import json
from lisa import FromRegions, parse_regions_file
from lisa.core.utils import LoadingBar
from collections import Counter
import pickle
from joblib import Parallel, delayed
import os
import fire

import tqdm
import torch
import pyro
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
from pyro.contrib.autoname import scope
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import torch.distributions.constraints as constraints
from pyro.infer.autoguide.initialization import init_to_mean

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


    def get_gene_model(self, gene_symbol):

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

        return RPModel(gene_symbol, **kwargs)


    def train_gene_models(self, gene_symbols,*,cell_topic_distribution, expr_matrix, read_depth, n_jobs = -1, 
        file_prefix='./', type = 'VI'):

        if type == 'VI':

            logging.info('Instantiating models ...')
            train_data = []
            for i, gene in enumerate(gene_symbols):
                try:
                    new_model = self.get_gene_model(gene)
                    train_data.append((
                        new_model,
                        dict(
                            weights = self.get_gene_weights(new_model, cell_topic_distribution),
                            expression = expr_matrix[:, i],
                            read_depth = read_depth
                        )
                    ))
                except IndexError:
                    logging.warning('Skipping ' + gene)

            def train_fn(model, fit_kwargs):

                try:
                    model.fit(**fit_kwargs).write_trace(file_prefix + model.gene_symbol + '_trace.pkl')
                    return model.gene_symbol + ': Success'

                except Exception as err:
                    return model.gene_symbol + ': ' + str(repr(err))
            
            logging.info('Training ...')
            return Parallel(n_jobs= n_jobs, verbose=10)([
                        delayed(train_fn)(model, params) for model, params in train_data
                    ])

        elif type == 'TransEffect':

            trans_model = TransEffectModel(gene_symbols).fit(
                cell_topic_weights= cell_topic_distribution,
                read_depth=read_depth,
                gene_expr=expr_matrix
            )

            trans_model.write_trace(file_prefix + 'transeffect_trace.pkl')

            return ['Done', ]

        else:
            raise ValueError('Model type {} does not exist'.format(str(type)))

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

    def sample_expression(self, region_weights, read_depth, n_samples = 500):

        expr_rate = np.exp(self.get_log_expr_rate(region_weights))

        return None

    def posterior_ISD(self, reg_state, motif_hits):

        #assert(motif_hits.shape[1] == reg_state.shape[0])
        assert(len(reg_state.shape) == 1)

        reg_state = reg_state[self.region_mask][np.newaxis, :]

        motif_hits = np.array(motif_hits[:, self.region_mask].todense())

        isd_mask = np.maximum(1 - motif_hits, 0)

        isd_states = np.vstack((reg_state, reg_state * isd_mask))

        rp_scores = np.exp(self.get_log_expr_rate(isd_states))

        return 1 - rp_scores[1:]/rp_scores[0]


class PyroModel(PyroModule):

    var_names = []

    def get_varnames(self):
        return self.var_names

    @classmethod
    def train(cls, *init_args, learning_rate = 0.01, iters = int(1e4), **init_kwargs):
        
        model = cls(*init_args, **init_kwargs)
        guide = AutoDiagonalNormal(model, init_loc_fn = init_to_mean)
        
        adam = pyro.optim.Adam({"lr": learning_rate})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        pyro.clear_param_store()
        for j in tqdm.tqdm(range(iters//100)):
            for i in range(100):
                loss = svi.step()
                
        return model, guide


class PyroRPVI(PyroModel):
    
    var_names = ['a','b','theta','dropout','logdistance']
    
    def __init__(self, gene_name, *,upstream_distances, downstream_distances, promoter_weights, 
            upstream_weights, downstream_weights, read_depth, gene_expr):
        super().__init__()
        self.upstream_distances = torch.tensor(upstream_distances)
        self.downstream_distances = torch.tensor(downstream_distances)
        self.upstream_weights = torch.tensor(upstream_weights)
        self.downstream_weights = torch.tensor(downstream_weights)
        self.promoter_weights = torch.tensor(promoter_weights)
        self.read_depth = torch.tensor(read_depth)
        self.name = gene_name
        self.N = promoter_weights.shape[0]
        self.gene_expr = torch.tensor(gene_expr)

    def get_varnames(self):
        return [self.name + '_' + varname for varname in self.var_names]
    
    def forward(self):

        def RP(weights, distances, d):
            return 1e4 * (weights * torch.pow(0.5, distances/(1e3 * d))).sum(-1)

        with pyro.plate(self.name +"_regions", 3):
            a = pyro.sample(self.name +"_a", dist.HalfNormal(12.))

        with pyro.plate(self.name +"_upstream-downstream", 2):

            d = torch.exp(pyro.sample(self.name +'_logdistance', dist.Normal(np.e, 2.)))

        b = pyro.sample(self.name +"_b", dist.Normal(-10.,3.))
        theta = pyro.sample(self.name +"_theta", dist.Gamma(2., 0.5))
        psi = pyro.sample(self.name +"_dropout", dist.Beta(1., 10.))

        with pyro.plate(self.name +"_data", self.N, subsample_size=64) as ind:

            expr_rate = a[0] * RP(self.upstream_weights.index_select(0, ind), self.upstream_distances, d[0])\
                + a[1] * RP(self.downstream_weights.index_select(0, ind), self.downstream_distances, d[1]) \
                + a[2] * 1e4 * self.promoter_weights.index_select(0, ind).sum(-1) \
                + b
            
            mu = torch.multiply(self.read_depth.index_select(0, ind), torch.exp(expr_rate))
            p = torch.minimum(mu / (mu + theta), torch.tensor([0.99999]))

            pyro.sample(self.name +'_obs', 
                        dist.ZeroInflatedNegativeBinomial(total_count=theta, probs=p, gate = psi),
                        obs= self.gene_expr.index_select(0, ind))


class PyroTransEffectModel(PyroModel):

    var_names = ['beta','b','theta','dropout']

    def __init__(self, gene_names, *, cell_topic_weights, read_depth, gene_expr):
        super().__init__()
        self.gene_names = gene_names
        self.K = cell_topic_weights.shape[-1]
        self.G = len(self.gene_names)
        self.N = len(cell_topic_weights)
        self.cell_topics = torch.tensor(cell_topic_weights)
        self.read_depth = torch.tensor(read_depth)
        self.gene_expr = torch.tensor(gene_expr)


    def forward(self):

        with pyro.plate("gene_weights", self.G):

            b = pyro.sample("b", dist.Normal(-10.,3.))
            theta = pyro.sample("theta", dist.Gamma(2., 0.5))
            psi = pyro.sample("dropout", dist.Beta(1., 10.))

            with pyro.plate("topic-gene_weights", self.K):
                beta = pyro.sample("beta", dist.Gamma(1., 5.))        
        
        with pyro.plate("gene", self.G) as gene:
            with pyro.plate("data", self.N, subsample_size=64) as ind:

                expr_rate = pyro.deterministic("rate", torch.matmul(self.cell_topics.index_select(0, ind), beta) + b)

                mu = torch.reshape(self.read_depth, (-1,1)).index_select(0, ind) * torch.exp(expr_rate)
                p = torch.minimum(mu / (mu + theta), torch.tensor([0.99999]))

                pyro.sample("obs",
                            dist.ZeroInflatedNegativeBinomial(total_count=theta, 
                                                              probs=p, gate = psi),
                            obs= self.gene_expr.index_select(0, ind))

class ExprModel:

    def sample_posterior(self, num_samples = 200, attempts = 5):

        for i in range(attempts):
            try:

                samples = Predictive(self.model, guide=self.guide, num_samples=num_samples,
                                    return_sites=self.model.get_varnames())()
                return {varname.split('_')[-1] : samples[varname].detach().numpy() for varname in self.model.get_varnames()}

            except ValueError:
                pass

        raise ValueError('Posterior contains improper values')

    def write_trace(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.trace, f)

    def load_trace(self, filename):

        with open(filename, 'rb') as f:
            self.trace = pickle.load(f)

        self.summarize_params()

    def summarize_params(self, metric = np.mean):
        for param in self.trace.keys():
            self.__setattr__(param, metric(self.trace[param], axis = 0))

    def get_log_expr_rate(self, *args, **kwargs):
        raise NotImplementedError()


class RPModel(ExprModel):

    def __init__(self, gene_symbol, *, region_mask, region_distances, upstream_mask, downstream_mask, promoter_mask, user_to_model_region_map):

        self.gene_symbol = gene_symbol
        self.region_mask = region_mask
        self.upstream_mask = upstream_mask
        self.downstream_mask = downstream_mask
        self.promoter_mask = promoter_mask
        self.region_distances = region_distances
        self.is_fit = False
        self.region_score_map = user_to_model_region_map
    

    def fit(self,*, weights, expression, read_depth, posterior_samples = 200):

        expression = np.array(expression).astype(np.float64)
        read_depth = np.array(read_depth).astype(np.int64)
        weights = np.array(weights).astype(np.float64)
        
        self.model, self.guide = PyroRPVI.train(
            self.gene_symbol,
            upstream_distances = self.region_distances[self.upstream_mask], 
            downstream_distances = self.region_distances[self.downstream_mask], 
            promoter_weights = weights[:, self.promoter_mask], 
            upstream_weights = weights[:, self.upstream_mask], 
            downstream_weights = weights[:, self.downstream_mask], 
            read_depth = read_depth,
            gene_expr = expression
        )

        self.trace = self.sample_posterior(posterior_samples)
        self.summarize_params()

        return self

    def get_rp_function(self):

        return RPModelPointEstimator(self.gene_symbol,
            a_up = self.a[0],
            a_down = self.a[1],
            a_promoter = self.a[2],
            distance_up = self.logdistance[0],
            distance_down = self.logdistance[1],
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

    def get_log_expr_rate(self, region_weights):
        return self.get_rp_function().get_log_expr_rate(region_weights)


class TransEffectModel(ExprModel):

    def __init__(self, gene_symbols):
        self.gene_symbols = gene_symbols

    def fit(self,*, cell_topic_weights, gene_expr, read_depth, posterior_samples = 200):

        self.model, self.guide = PyroTransEffectModel.train(
            self.gene_symbols,
            cell_topic_weights = cell_topic_weights, 
            read_depth = read_depth,
            gene_expr = gene_expr
        )

        self.trace = self.sample_posterior(posterior_samples)
        self.summarize_params()

        return self

    def get_log_expr_rate(self, cell_topic_weights):
        return np.dot(cell_topic_weights, self.beta) + self.b


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