
import os
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.optim import Adam
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
from pyro.nn import PyroModule
import numpy as np
import torch.distributions.constraints as constraints
import fire
from pyro.infer import Predictive
import logging
import pickle

logging.basicConfig(level = logging.INFO)
logger = logging.Logger('ExprLDA')
logger.setLevel(logging.INFO)

class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, num_genes, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, num_genes, bias = False)
        self.bn = nn.BatchNorm1d(num_genes)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)

class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        theta_loc = self.bnmu(self.fcmu(h))
        theta_scale = self.bnlv(self.fclv(h))
        theta_scale = (0.5 * theta_scale).exp()  # Enforces positivity
        return theta_loc, theta_scale
        

class scVLCM_Estimator:

    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def softmax(x):
        return np.exp(x)/np.exp(x).sum(-1, keepdims = True)

    def predict(self, theta):
        assert(np.isclose(theta.sum(-1), 1).all()), 'Theta latent var must be composition'

        activation = (np.dot(theta, self.beta) - self.bn_mean)/np.sqrt(self.bn_var)

        return self.softmax(self.gamma * activation + self.bias)


class ExpressionModel(nn.Module):

    guide_vars = ['gamma_a','gamma_b','dropout_a','dropout_b']
    variational_vars = ['theta','beta', 'bias','gamma','bn_mean','bn_var']

    @classmethod
    def get_estimator_from_trace(cls, prefix):
        
        trace = {}
        for suffix in cls.variational_vars:
            try:
                filename = prefix + '_{}.npy'.format(suffix)
                m = np.load(filename)
                trace[suffix] = m
            except FileNotFoundError:
                raise Exception('Cannot make model estimator, missing file: {}'.format(filename))

        return scVLCM_Estimator(**trace)

    def __init__(self, num_genes,num_topics = 15, initial_counts = 50, 
        dropout = 0.2, hidden = 128, use_cuda = True):
        super().__init__()

        assert(isinstance(initial_counts, int))

        a = initial_counts/num_topics
        self.prior_mu = 0
        self.prior_std = np.sqrt(1/a * (1-2/num_topics) + 1/(num_topics * a))

        self.num_genes = num_genes
        self.num_topics = num_topics
        self.decoder = Decoder(self.num_genes, num_topics, dropout)
        self.encoder = Encoder(self.num_genes, num_topics, hidden, dropout)

        self.use_cuda = torch.cuda.is_available() and use_cuda
        logging.info('Using CUDA: {}'.format(str(self.use_cuda)))
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        self.to(self.device)
        self.max_prob = torch.tensor([0.99999], requires_grad = False).to(self.device)

    def get_estimator(self):
        return scVLCM_Estimator(**{varname : getattr(self, varname) for varname in self.variational_vars})

    def write_trace(self, prefix):
        logging.info('Writing results ...')
        for varname in self.variational_vars + self.guide_vars:
            np.save(prefix + '_{}.npy'.format(varname), getattr(self, varname))

    def model(self, raw_expr, encoded_expr, read_depth):

        pyro.module("decoder", self.decoder)

        with pyro.plate("genes", self.num_genes):

            dispersion = pyro.sample("dispersion", dist.Gamma(torch.tensor(2.).to(self.device), torch.tensor(0.5).to(self.device)) )
            psi = pyro.sample("dropout", dist.Beta(torch.tensor(1.).to(self.device), torch.tensor(10.).to(self.device)) )
        
        #pyro.module("decoder", self.decoder)
        with pyro.plate("cells", encoded_expr.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution

            theta_loc = self.prior_mu * encoded_expr.new_ones((encoded_expr.shape[0], self.num_topics))
            theta_scale = self.prior_std * encoded_expr.new_ones((encoded_expr.shape[0], self.num_topics))
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta/theta.sum(-1, keepdim = True)
            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            expr_rate = pyro.deterministic("expr_rate", self.decoder(theta))

            mu = torch.multiply(read_depth, expr_rate)
            p = torch.minimum(mu / (mu + dispersion), self.max_prob)

            pyro.sample('obs', 
                        dist.ZeroInflatedNegativeBinomial(total_count=dispersion, probs=p, gate = psi).to_event(1),
                        obs= raw_expr)


    def guide(self, raw_expr, encoded_expr, read_depth):

        pyro.module("encoder", self.encoder)

        with pyro.plate("genes", self.num_genes):
            
            gamma_a = pyro.param("gamma_a", torch.tensor(2., device = self.device), constraint = constraints.positive)
            gamma_b = pyro.param("gamma_b", torch.tensor(0.5, device = self.device), constraint = constraints.positive)
            dispersion = pyro.sample("dispersion", dist.Gamma(gamma_a, gamma_b))

            dropout_a = pyro.param("dropout_a", torch.tensor(1., device = self.device), constraint = constraints.positive)
            dropout_b = pyro.param("dropout_b", torch.tensor(10., device = self.device), constraint = constraints.positive)
            dropout = pyro.sample("dropout", dist.Beta(dropout_a, dropout_b))          


        with pyro.plate("cells", encoded_expr.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the encoder network outputs
            theta_loc, theta_scale = self.encoder(encoded_expr)

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))


    def get_latent_MAP(self, encoded_expr):
        theta_loc, theta_scale = self.encoder(encoded_expr)
        return np.exp(theta_loc.cpu().detach().numpy())


    def summarize_posterior(self, raw_expr, encoded_expr, read_depth):

        logging.info('Sampling posterior ...')
        
        #Local vars
        latent_vars = []
        for i,batch in enumerate(self.epoch_batch(raw_expr, encoded_expr, read_depth, batch_size = 512)):
            latent_vars.append(self.get_latent_MAP(batch[1]))

        self.theta = np.vstack(latent_vars)
        #Decoder vars
        self.beta = self.get_beta()
        self.gamma = self.get_gamma()
        self.bias = self.get_bias()
        self.bn_mean = self.get_bn_mean()
        self.bn_var = self.get_bn_var()

        #Guide vars
        for guide_var in self.guide_vars:
            self.__setattr__(guide_var, pyro.param(guide_var).item().cpy().detach().numpy())
        
        return self

    def epoch_batch(self, *data, batch_size = 32):

        N = len(data[0])
        num_batches = N//batch_size + int(N % batch_size > 0)

        for i in range(num_batches):
            batch_start, batch_end = (i * batch_size, (i + 1) * batch_size)

            yield list(map(lambda x : x[batch_start:batch_end], data))

    def train(self, *, raw_expr, encoded_expr, num_epochs = 100, 
            batch_size = 32, learning_rate = 1e-3, posterior_samples = 20):

        logging.info('Validating data ...')

        assert(raw_expr.shape == encoded_expr.shape)
        
        read_depth = torch.tensor(raw_expr.sum(-1)[:, np.newaxis]).to(self.device)
        raw_expr = torch.tensor(raw_expr).to(self.device)
        encoded_expr = torch.tensor(encoded_expr).to(self.device)

        logging.info('Initializing model ...')

        pyro.clear_param_store()
        adam_params = {"lr": 1e-3}
        optimizer = Adam(adam_params)
        svi = SVI(self.model, self.guide, optimizer, loss=TraceMeanField_ELBO())

        logging.info('Training ...')
        num_batches = read_depth.shape[0]//batch_size
        bar = trange(num_epochs)

        try:
            for epoch in bar:
                running_loss = 0.0
                for batch in self.epoch_batch(raw_expr, encoded_expr, read_depth, batch_size = batch_size):
                    loss = svi.step(*batch)
                    running_loss += loss / batch_size

                bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
        except KeyboardInterrupt:
            logging.error('Interrupted training.')

        self.summarize_posterior(raw_expr, encoded_expr, read_depth)

        return self

    def get_beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T.numpy()

    def get_gamma(self):
        return self.decoder.bn.weight.cpu().detach().numpy()
    
    def get_bias(self):
        return self.decoder.bn.bias.cpu().detach().numpy()

    def get_bn_mean(self):
        return self.decoder.bn.running_mean.cpu().detach().numpy()

    def get_bn_var(self):
        return self.decoder.bn.running_var.cpu().detach().numpy()

def main(*,raw_expr, encoded_expr, save_file, topics = 32, hidden_layers=128,
    learning_rate = 1e-3, epochs = 100, batch_size = 32, initial_counts = 20):

    raw_expr = np.load(raw_expr)
    encoded_expr = np.load(encoded_expr)

    seed = 2556
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)    

    model = ExpressionModel(raw_expr.shape[-1], num_topics = topics, dropout = 0.2, 
        hidden = hidden_layers, initial_counts = initial_counts)

    model.train(
        raw_expr = raw_expr,
        encoded_expr = encoded_expr,
        num_epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate
    )

    model.write_trace(save_file)

if __name__ == "__main__":
    fire.Fire(main)