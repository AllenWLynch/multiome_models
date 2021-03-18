
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
        self.beta = nn.Linear(num_topics, num_genes)
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


class ProdLDA(PyroModule):

    local_vars = ['theta','expr_rate']
    global_vars = ['dispersion','dropout']

    var_names = local_vars + global_vars

    def __init__(self, num_genes,num_topics = 15, initial_counts = 50, dropout = 0.2, hidden = 128):
        super().__init__()

        assert(isinstance(initial_counts, int))

        a = initial_counts/num_topics
        self.prior_mu = 0
        self.prior_std = np.sqrt(1/a * (1-2/num_topics) + 1/(num_topics * a))

        self.num_genes = num_genes
        self.num_topics = num_topics
        self.decoder = Decoder(self.num_genes, num_topics, dropout)
        self.encoder = Encoder(self.num_genes, num_topics, hidden, dropout)

    def write_trace(self, filename):
        logging.info('Writing trace ...')
        with open(filename, 'wb') as f:
            pickle.dump(self.trace, f)

    def load_trace(self, filename):

        with open(filename, 'rb') as f:
            self.trace = pickle.load(f)

        self.summarize_params()

    def summarize_params(self, metric = np.mean):
        logging.info('Summarizing params ...')

        for param in self.var_names:
            self.__setattr__(param, metric(self.trace[param], axis = 0))

    def get_log_expr_rate(self, *args, **kwargs):
        raise NotImplementedError()

    def model(self, raw_expr, encoded_expr, read_depth):

        pyro.module("decoder", self.decoder)

        with pyro.plate("genes", self.num_genes):

            dispersion = pyro.sample("dispersion", dist.Gamma(2., 0.5))
            psi = pyro.sample("dropout", dist.Beta(1., 10.))
        
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
            p = torch.minimum(mu / (mu + dispersion), torch.tensor([0.99999]))

            pyro.sample('obs', 
                        dist.ZeroInflatedNegativeBinomial(total_count=dispersion, probs=p, gate = psi).to_event(1),
                        obs= raw_expr)


    def guide(self, raw_expr, encoded_expr, read_depth):

        pyro.module("encoder", self.encoder)

        with pyro.plate("genes", self.num_genes):
            
            gamma_a = pyro.param("gamma_a", torch.tensor(2.), constraint = constraints.positive)
            gamma_b = pyro.param("gamma_b", torch.tensor(0.5), constraint = constraints.positive)
            dispersion = pyro.sample("dispersion", dist.Gamma(gamma_a, gamma_b))

            dropout_a = pyro.param("dropout_a", torch.tensor(1.), constraint = constraints.positive)
            dropout_b = pyro.param("dropout_b", torch.tensor(10.), constraint = constraints.positive)
            dropout = pyro.sample("dropout", dist.Beta(dropout_a, dropout_b))           


        with pyro.plate("cells", encoded_expr.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the encoder network outputs
            theta_loc, theta_scale = self.encoder(encoded_expr)

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))

    def sample_posterior(self, raw_expr, encoded_expr, read_depth, 
        num_samples = 200, attempts = 5):

        logging.info('Sampling posterior ...')

        self.posterior_predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples,
                    return_sites=self.var_names)
                    
        trace = {}
        for i,batch in enumerate(self.epoch_batch(raw_expr, encoded_expr, read_depth, batch_size = 512)):
            
            samples = self.posterior_predictive(*batch)
            
            if i == 0:
                for varname in self.global_vars:
                    new_samples = samples[varname].detach().numpy()
                    trace[varname] = new_samples
            
            for varname in self.local_vars:
                new_samples = samples[varname].detach().numpy()
                
                if not varname in trace:
                    trace[varname] = []
                trace[varname].append(new_samples)

            logging.info('Done {} batches.'.format(str(i+1)))

        for varname in self.local_vars:
            trace[varname] = np.concatenate(trace[varname], axis = 1)

        trace['beta'] = self.get_beta()
        trace['bias'] = self.get_bias()

        return trace

    @staticmethod
    def softmax(x):
        return np.exp(x)/np.exp(x).sum(-1, keepdims = True)

    def predict(self, theta):
        return self.softmax(np.dot(self.theta, self.trace['beta']) + self.trace['bias'][np.newaxis, :])

    def epoch_batch(self, *data, batch_size = 32):

        N = len(data[0])
        num_batches = N//batch_size + int(N % batch_size > 0)

        for i in range(num_batches):
            batch_start, batch_end = (i * batch_size, (i + 1) * batch_size)

            yield list(map(lambda x : x[batch_start:batch_end], data))

    def train(self, *, raw_expr, encoded_expr, read_depth, num_epochs = 100, 
            batch_size = 32, learning_rate = 1e-3, posterior_samples = 20):

        logging.info('Initializing model ...')

        pyro.clear_param_store()
        adam_params = {"lr": 1e-3}
        optimizer = Adam(adam_params)
        svi = SVI(self.model, self.guide, optimizer, loss=TraceMeanField_ELBO())

        raw_expr = torch.tensor(raw_expr)
        encoded_expr = torch.tensor(encoded_expr)
        read_depth = torch.tensor(read_depth[:, np.newaxis])

        num_batches = read_depth.shape[0]//batch_size
        bar = trange(num_epochs)

        logging.info('Training ...')
        for epoch in bar:
            running_loss = 0.0
            for batch in self.epoch_batch(raw_expr, encoded_expr, read_depth, batch_size = batch_size):
                loss = svi.step(*batch)
                running_loss += loss / batch_size

            bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))

        self.trace = self.sample_posterior(raw_expr, encoded_expr, read_depth, 
                num_samples = posterior_samples)
        self.summarize_params()

        return self

    def get_beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T.numpy()

    def get_bias(self):
        return self.decoder.beta.bias.cpu().detach().T.numpy()

def main(*,raw_expr, encoded_expr, read_depth, save_file, topics = 32, hidden_layers=128,
    learning_rate = 1e-3, epochs = 100, batch_size = 32, posterior_samples = 20):

    raw_expr = np.load(raw_expr)
    encoded_expr = np.load(encoded_expr)
    read_depth = np.load(read_depth)

    assert(raw_expr.shape == encoded_expr.shape)
    assert(raw_expr.shape[0] == read_depth.shape[0])
    assert(len(read_depth.shape) == 1)

    rna_topic = ProdLDA(raw_expr.shape[-1], num_topics = topics, dropout = 0.2, hidden = hidden_layers)
    
    rna_topic.train(
        raw_expr = raw_expr,
        encoded_expr = encoded_expr,
        read_depth = read_depth,
        num_epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        posterior_samples = posterior_samples,
    )

    rna_topic.write_trace(save_file)

if __name__ == "__main__":
    fire.Fire(main)


