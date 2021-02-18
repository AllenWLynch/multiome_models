from sklearn.metrics import pairwise_distances_chunked
from functools import partial
import sys
import numpy as np
from scipy import sparse


def reduce_distance_matrix_func(distances,start,*,ka,neighborhood_size):
    neighbor_coors = np.argpartition(distances, kth = neighborhood_size*ka+1)[:,:neighborhood_size*ka+1]
    neighbor_distances = np.take_along_axis(distances, neighbor_coors, axis = 1)
    kernel_width = np.sort(neighbor_distances, axis = 1)[:,ka][:,np.newaxis]

    return neighbor_coors, neighbor_distances, kernel_width

def get_matrix_params(embeddings,*,ka,neighborhood_size,
    metric = 'euclidean', n_jobs = 1, working_memory = None):

    reduce_function = partial(reduce_distance_matrix_func, ka = ka, 
        neighborhood_size = neighborhood_size)

    neighbor_coors, neighbor_distances, kernel_width = list(map(np.concatenate, 
        zip(*pairwise_distances_chunked(embeddings, reduce_func = reduce_function,
            metric = metric, n_jobs = n_jobs, working_memory = working_memory)
        )
    ))

    return neighbor_coors, neighbor_distances, kernel_width

def build_affinity_matrix(neighbor_coors, neighbor_distances, kernel_width,*,
    neighborhood_size,ka):

    i = np.repeat(np.arange(neighbor_distances.shape[0]), neighborhood_size*ka+1)
    j = neighbor_coors.reshape(-1)

    values = neighbor_distances.reshape(-1)

    kernel_width_reps = np.repeat(kernel_width, neighborhood_size*ka+1)

    affinity = np.exp(-1*np.square(values/kernel_width_reps))

    affinity_matrix = sparse.csr_matrix((affinity, (i,j)))

    affinity_matrix = (affinity_matrix + affinity_matrix.transpose().tocsr())

    return affinity_matrix


def set_diagonal(affinity_matrix, value):
    affinity_matrix = affinity_matrix.tolil()
    affinity_matrix.setdiag(value)
    return affinity_matrix.tocsr()


def make_markov_matrix(affinity_matrix):
    inverse_rowsums = sparse.diags(1/np.array(affinity_matrix.sum(axis = 1)).reshape(-1)).tocsr()
    markov_matrix = inverse_rowsums.dot(affinity_matrix)
    return markov_matrix

def diffuse_matrix(markov_matrix, diffusion_time):
    return markov_matrix**diffusion_time

def calc_markov_diffusion_matrix(embeddings, diffusion_time = 3, metric = 'euclidean', ka = 5,
                                leave_self_out = False, neighborhood_size = 3, n_jobs=1, working_memory=None):
    
    affinity_params = get_matrix_params(embeddings, ka = ka, neighborhood_size = neighborhood_size,
        n_jobs=n_jobs, working_memory=working_memory, metric=metric)

    affinity_matrix = build_affinity_matrix(*affinity_params,neighborhood_size=neighborhood_size,ka=ka)

    if leave_self_out:
        affinity_matrix = set_diagonal(affinity_matrix, 0.0)

    markov_matrix = make_markov_matrix(affinity_matrix)

    diffusion_matrix = diffuse_matrix(markov_matrix, diffusion_time)

    return diffusion_matrix

def impute(feature_matrix, diffusion_matrix):
    imputed_data = diffusion_matrix.dot(feature_matrix)
    return imputed_data

def get_imputation_error(impute_values,*, observed_embeddings, expected_embeddings, diffusion_time = 3, metric = 'euclidean', ka = 5, neighborhood_size = 3, n_jobs=1, working_memory=None):

    markov_kwargs = dict(diffusion_time=diffusion_time, metric=metric,
        ka=ka, leave_self_out=True,neighborhood_size=neighborhood_size, n_jobs=n_jobs, working_memory=working_memory)

    observed_markov_matrix = calc_markov_diffusion_matrix(observed_embeddings, **markov_kwargs)
    observed_imputation = impute(impute_values, observed_markov_matrix)
    
    expected_markov_matrix = calc_markov_diffusion_matrix(expected_embeddings, **markov_kwargs)
    expected_imputation = impute(impute_values, expected_markov_matrix)

    return np.sqrt(np.sum(np.square(observed_imputation - expected_imputation), axis = 1)), observed_imputation, expected_imputation