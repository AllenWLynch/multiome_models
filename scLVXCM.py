
import os
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.optim import Adam
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm
from pyro.nn import PyroModule
import numpy as np
import torch.distributions.constraints as constraints
import fire
from pyro.infer import Predictive
import logging
import pickle
from pyro import poutine
from pyro.infer.util import torch_item

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
        self.fc = nn.Linear(num_topics, num_genes, bias = False)
        self.bn2 = nn.BatchNorm1d(num_genes)

    def forward(self, latent_composition):
        inputs = self.drop(latent_composition)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(self.drop(inputs))), dim=1), self.bn2(self.fc(inputs))

class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, num_genes, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(num_genes + 1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.fcrd = nn.Linear(hidden, 2)
        self.bnmu = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnrd = nn.BatchNorm1d(2)

    def forward(self, inputs):
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        theta_loc = self.bnmu(self.fcmu(h))
        theta_scale = self.bnlv(self.fclv(h))
        theta_scale = F.softplus(theta_scale) #(0.5 * theta_scale).exp()  # Enforces positivity
        
        rd = self.bnrd(self.fcrd(h))
        rd_loc = rd[:,0]
        rd_scale = F.softplus(rd[:,1]) #(0.5 * rd[:,1]).exp()

        return theta_loc, theta_scale, rd_loc, rd_scale

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

    guide_vars = ['dispersion']
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
        #self.max_scale = torch.tensor([1e6], requires_grad = False).to(self.device)

    def get_estimator(self):
        return scVLCM_Estimator(**{varname : getattr(self, varname) for varname in self.variational_vars})

    def write_trace(self, prefix):
        logging.info('Writing results ...')
        for varname in self.variational_vars + self.guide_vars:
            np.save(prefix + '_{}.npy'.format(varname), getattr(self, varname))

    def model(self, raw_expr, encoded_expr, read_depth):

        pyro.module("decoder", self.decoder)

        dispersion = pyro.param("dispersion", torch.tensor(5.).to(self.device) * torch.ones(self.num_genes).to(self.device), 
            constraint = constraints.positive)
        
        with pyro.plate("cells", encoded_expr.shape[0]):

            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution
            theta_loc = self.prior_mu * encoded_expr.new_ones((encoded_expr.shape[0], self.num_topics))
            theta_scale = self.prior_std * encoded_expr.new_ones((encoded_expr.shape[0], self.num_topics))
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta/theta.sum(-1, keepdim = True)

            read_scale = pyro.sample(
                'read_depth', dist.LogNormal(torch.log(read_depth), 1.).to_event(1)
            )

            #read_scale = torch.minimum(read_scale, self.max_scale)
            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            expr_rate, dropout = self.decoder(theta)

            mu = torch.multiply(read_scale, expr_rate)
            p = torch.minimum(mu / (mu + dispersion), self.max_prob)

            pyro.sample('obs', 
                        dist.ZeroInflatedNegativeBinomial(total_count=dispersion, probs=p, gate_logits=dropout).to_event(1),
                        obs= raw_expr)


    def guide(self, raw_expr, encoded_expr, read_depth):

        pyro.module("encoder", self.encoder)

        with pyro.plate("cells", encoded_expr.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the encoder network outputs
            theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(encoded_expr)

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))

            read_depth = pyro.sample(
                "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1))


    def epoch_batch(self, *data, batch_size = 32, bar = True):

        N = len(data[0])
        num_batches = N//batch_size + int(N % batch_size > 0)

        for i in tqdm(range(num_batches), desc = 'Epoch progress') if bar is True else range(num_batches):
            batch_start, batch_end = (i * batch_size, (i + 1) * batch_size)

            yield list(map(lambda x : x[batch_start:batch_end], data))


    def evaluate(self, raw_expr, encoded_expr, read_depth):
        
        batch_logp = []
        for batch in self.epoch_batch(raw_expr, encoded_expr, read_depth, batch_size = 512, bar = False):

            with torch.no_grad():
                log_prob = torch_item(self.loss.loss(self.model, self.guide, *batch))

            batch_logp.append(log_prob)

        return np.array(batch_logp).sum() #loss is negative log-likelihood

    def beta_l1_loss(self):
        return -dist.Laplace(0., 1.).log_prob(self.decoder.beta.weight).sum()

    def custom_step(self, *batch):

        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss.differentiable_loss(self.model, self.guide, *batch) + self.beta_l1_loss()
            
            params = set(site["value"].unconstrained()
                            for site in param_capture.trace.nodes.values())

        loss.backward()
        self.optimizer(params)
        pyro.infer.util.zero_grads(params)

        return torch_item(loss)


    def train(self, *, raw_expr, encoded_expr, num_epochs = 100, 
            batch_size = 32, learning_rate = 1e-3, eval_every = 10, test_proportion = 0.05,
            use_l1 = False, l1_lam = 0):

        seed = 2556
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)

        pyro.clear_param_store()
        logging.info('Validating data ...')

        assert(raw_expr.shape == encoded_expr.shape)
        read_depth = raw_expr.sum(-1)[:, np.newaxis]

        encoded_expr = np.hstack([encoded_expr, np.log(read_depth)])

        read_depth = torch.tensor(read_depth).to(self.device)
        raw_expr = torch.tensor(raw_expr).to(self.device)
        encoded_expr = torch.tensor(encoded_expr).to(self.device)

        logging.info('Initializing model ...')

        self.optimizer = Adam({"lr": 1e-3})
        self.loss = TraceMeanField_ELBO()
        
        if not use_l1:
            logging.info('No L1 regularization.')
            svi = SVI(self.model, self.guide, self.optimizer, loss=self.loss)

        test_set = np.random.rand(read_depth.shape[0]) < test_proportion
        train_set = ~test_set

        logging.info("Training with {} cells, testing with {}.".format(str(train_set.sum()), str(test_set.sum())))
        logging.info('Training ...')

        try:
            for epoch in range(1, num_epochs + 1):
                running_loss = 0.0
                for batch in self.epoch_batch(raw_expr[train_set], encoded_expr[train_set], read_depth[train_set], batch_size = batch_size):
                    if use_l1:
                        loss = self.custom_step(*batch)
                    else:
                        loss = svi.step(*batch)

                    running_loss += loss / batch_size

                logging.info('Done epoch {}/{}. Training loss: {:.3e}'.format(str(epoch), str(num_epochs), running_loss))

                if (epoch % eval_every == 0 or epoch == num_epochs) and test_set.sum() > 0:
                    test_logp = self.evaluate(raw_expr[test_set], encoded_expr[test_set], read_depth[test_set])
                    logging.info('Test logp: {:.4e}'.format(test_logp))

        except KeyboardInterrupt:
            logging.error('Interrupted training.')

        self.summarize_posterior(raw_expr, encoded_expr, read_depth)

        return self

    def get_latent_MAP(self, encoded_expr):
        theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(encoded_expr)
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
        self.dispersion = self.get_dispersion()
        
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

    def get_dispersion(self):
        return pyro.param("dispersion").cpu().detach().numpy()

def main(*,raw_expr, encoded_expr, save_file, topics = 32, hidden_layers=128,
    learning_rate = 1e-3, epochs = 100, batch_size = 32, initial_counts = 20, eval_every = 10, 
    test_proportion = 0.05, use_l1 = False, dropout = 0.2):

    raw_expr = np.load(raw_expr).astype(np.float32)
    encoded_expr = np.load(encoded_expr).astype(np.float32)

    raw_expr = raw_expr
    encoded_expr = encoded_expr

    if use_l1:
        dropout = 0.0

    model = ExpressionModel(raw_expr.shape[-1], num_topics = topics, dropout = dropout, 
        hidden = hidden_layers, initial_counts = initial_counts)

    model.train(
        raw_expr = raw_expr,
        encoded_expr = encoded_expr,
        num_epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        eval_every = eval_every,
        test_proportion = test_proportion,
        use_l1 = use_l1
    )

    model.write_trace(save_file)

if __name__ == "__main__":
    fire.Fire(main)