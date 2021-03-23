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
from scipy.sparse import isspmatrix, load_npz

logging.basicConfig(level = logging.INFO)
logger = logging.Logger('ExprLDA')
logger.setLevel(logging.INFO)

class DANEncoder(nn.Module):

    def __init__(self, num_peaks, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_peaks, hidden, padding_idx=0)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics)  # to avoid component collapse

    def forward(self, idx, read_depth):

        embeddings = self.drop(self.embedding(idx)) # N, T, D
        
        ave_embeddings = embeddings.sum(1)/read_depth

        h = F.softplus(self.fc1(ave_embeddings))
        h = F.softplus(self.fc2(h))
        h = self.drop2(h)

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


class AccessibilityModel(nn.Module):

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

    def __init__(self, num_peaks, num_topics = 15, initial_counts = 20, 
        dropout = 0.2, hidden = 128, use_cuda = True):
        super().__init__()

        assert(isinstance(initial_counts, int))

        a = initial_counts/num_topics
        self.prior_mu = 0
        self.prior_std = np.sqrt(1/a * (1-2/num_topics) + 1/(num_topics * a))

        self.num_peaks = num_peaks
        self.num_topics = num_topics
        self.decoder = Decoder(self.num_peaks, num_topics, dropout)
        self.encoder = DANEncoder(self.num_peaks, num_topics, hidden, dropout)

        self.use_cuda = torch.cuda.is_available() and use_cuda
        logging.info('Using CUDA: {}'.format(str(self.use_cuda)))
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        self.to(self.device)

    def get_estimator(self):
        return scVLCM_Estimator(**{varname : getattr(self, varname) for varname in self.variational_vars})

    def write_trace(self, prefix):
        logging.info('Writing results ...')
        for varname in self.variational_vars:
            np.save(prefix + '_{}.npy'.format(varname), getattr(self, varname))

    def model(self, peak_idx, read_depth, onehot_obs = None):

        pyro.module("decoder", self.decoder)
        
        #pyro.module("decoder", self.decoder)
        with pyro.plate("cells", peak_idx.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution

            theta_loc = self.prior_mu * peak_idx.new_ones((peak_idx.shape[0], self.num_topics))
            theta_scale = self.prior_std * peak_idx.new_ones((peak_idx.shape[0], self.num_topics))
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta/theta.sum(-1, keepdim = True)
            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            peak_probs = self.decoder(theta)
            
            pyro.sample(
                'obs',
                dist.Multinomial(total_count = read_depth if onehot_obs is None else 1, probs = peak_probs).to_event(1),
                obs=onehot_obs
            )

    def guide(self, peak_idx, read_depth, onehot_obs = None):

        pyro.module("encoder", self.encoder)

        with pyro.plate("cells", peak_idx.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the encoder network outputs
            theta_loc, theta_scale = self.encoder(peak_idx, read_depth)

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))

    def get_latent_MAP(self, idx, read_depth):
        theta_loc, theta_scale = self.encoder(idx, read_depth)
        return np.exp(theta_loc.cpu().detach().numpy())

    def summarize_posterior(self, peak_idx, read_depth):

        logging.info('Sampling posterior ...')

        #Local vars    
        latent_vars = []
        for i,(idx, rd, _) in enumerate(self.epoch_batch(peak_idx, read_depth, batch_size = 512)):
            latent_vars.append(self.get_latent_MAP(idx, rd))

        self.theta = np.vstack(latent_vars)
        #Decoder vars
        self.beta = self.get_beta()
        self.gamma = self.get_gamma()
        self.bias = self.get_bias()
        self.bn_mean = self.get_bn_mean()
        self.bn_var = self.get_bn_var()

        return self

    def get_onehot_tensor(self, idx):
        return torch.zeros(idx.shape[0], self.num_peaks).scatter_(1, idx, 1).to(self.device)

    def epoch_batch(self, peak_idx, read_depth, batch_size = 32):

        N = peak_idx.shape[0]
        num_batches = N//batch_size + int(N % batch_size > 0)

        for i in range(num_batches):
            batch_start, batch_end = (i * batch_size, (i + 1) * batch_size)

            idx_batch = peak_idx[batch_start : batch_end]
            yield idx_batch, read_depth[batch_start:batch_end], self.get_onehot_tensor(idx_batch)

    def get_padded_idx_matrix(self, accessibility_matrix):
        
        assert(isspmatrix(accessibility_matrix))
        assert(accessibility_matrix.shape[1] <= self.num_peaks)

        X = accessibility_matrix.tolil().rows

        dense_matrix = []
        for row in X:
            if len(row) == self.num_peaks:
                dense_matrix.append(np.array(row)[np.newaxis, :])
            else:
                dense_matrix.append(np.concatenate([np.array(row), np.zeros(self.num_peaks - len(row))])[np.newaxis, :]) #0-pad tail to "width"

        dense_matrix = np.vstack(dense_matrix)
        
        return dense_matrix.astype(np.int64)


    def train(self, *, accessibility_matrix, num_epochs = 125, 
            batch_size = 32, learning_rate = 1e-3):

        logging.info('Validating data ...')

        peak_idx_matrix = self.get_padded_idx_matrix(accessibility_matrix)
        read_depth = (peak_idx_matrix > 0).sum(-1, keepdims = True).astype(np.int64)

        peak_idx_matrix = torch.tensor(peak_idx_matrix).to(self.device)
        read_depth = torch.tensor(read_depth).to(self.device)
        
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
                for batch in self.epoch_batch(peak_idx_matrix, read_depth, batch_size = batch_size):
                    loss = svi.step(*batch)
                    running_loss += loss / batch_size

                bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))

        except KeyboardInterrupt:
            logging.error('Interrupted training.')

        self.summarize_posterior(peak_idx_matrix, read_depth)

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


def main(*,raw_expr, encoded_expr, read_depth, save_file, topics = 32, hidden_layers=128,
    learning_rate = 1e-3, epochs = 100, batch_size = 32, initial_counts = 50):

    accessibility_matrix = load_npz(accessibility_matrix)

    seed = 2556
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    
    model = AccessibilityModel(accessibility_matrix.shape[1], num_topics = topics, dropout = 0.2, 
        hidden = hidden_layers, initial_counts = initial_counts)

    model.train(
        accessibility_matrix=accessibility_matrix,
        num_epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
    )

    model.write_trace(save_file)

if __name__ == "__main__":
    fire.Fire(main)