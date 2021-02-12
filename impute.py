from scipy.spatial.distance import cdist
from scipy import sparse
import sys
import numpy as np
import argparse
import os

def calc_markov_diffusion_matrix(embeddings, diffusion_time = 3, distance_metric = 'euclidean', ka = 5,
                                leave_self_out = False, neighborhood_size = 3):
    
    print('Calculating distances ...', file = sys.stderr)
    distances = cdist(embeddings, embeddings, distance_metric)

    print('Constructing local neighborhoods ...', file = sys.stderr)

    neighbor_coors = np.argpartition(distances, kth = neighborhood_size*ka+1)[:,:neighborhood_size*ka+1]
    neighbor_distances = np.take_along_axis(distances, neighbor_coors, axis = 1)
    del distances

    print('Converting to Markov matrix ...', file = sys.stderr)
    kernel_width = np.sort(neighbor_distances, axis = 1)[:,ka+1][:,np.newaxis]

    i = np.repeat(np.arange(neighbor_distances.shape[0]), neighborhood_size*ka+1)
    j = neighbor_coors.reshape(-1)

    values = neighbor_distances.reshape(-1)

    kernel_width_reps = np.repeat(kernel_width, neighborhood_size*ka+1)

    affinity = np.exp(-1*np.square(values/kernel_width_reps))

    affinity_matrix = sparse.csr_matrix((affinity, (i,j)))

    affinity_matrix_symm = (affinity_matrix + affinity_matrix.transpose().tocsr())
    
    if leave_self_out:
        leave_self_out_affinity = affinity_matrix_symm.tolil()
        leave_self_out_affinity.setdiag(0)
        affinity_matrix_symm = leave_self_out_affinity.tocsr()
    

    inverse_rowsums = sparse.diags(1/np.array(affinity_matrix_symm.sum(axis = 1)).reshape(-1)).tocsr()

    markov_matrix = inverse_rowsums.dot(affinity_matrix_symm)

    print('Diffusing ...', file = sys.stderr)
    diffusion_matrix = markov_matrix**diffusion_time

    percent_sparsity = 100*(1 - diffusion_matrix.size/(diffusion_matrix.shape[0]**2))
    print('Computed diffusion matrix with {}% sparsity.'.format(str(percent_sparsity)), file = sys.stderr)
    
    return diffusion_matrix

def impute(*,feature_matrix, diffusion_matrix):
    print('Imputing values ...', file = sys.stderr)
    imputed_data = diffusion_matrix.dot(feature_matrix)
    print('Done!', file = sys.stderr)
    return imputed_data


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    files_group = parser.add_argument_group('files')
    files_group.add_argument('--embeddings', type = str, required=True, help='Dimensionality-reduced features to calculate inter-sample distances')
    files_group.add_argument('--raw_data', type = str, required=True, help = 'Noisy features to impute using markov matrix')
    files_group.add_argument('-o','--output_prefix',type = str)
    params_group = parser.add_argument_group('params')
    params_group.add_argument('-d','--diffusion_time', type = int, default=4)
    params_group.add_argument('-ka',type = int, default=4)
    params_group.add_argument('--neighborhood_size', type = int, default=4)
    params_group.add_argument('--distance_metric', type = str, default='euclidean')
    params_group.add_argument('--save_markov_matrix', action='store_true')
    args = parser.parse_args()

    raw_data = np.load(args.raw_data)
    embeddings = np.load(args.embeddings)

    markov_matrix = calc_markov_diffusion_matrix(embeddings, diffusion_time = args.diffusion_time, distance_metric = args.distance_metric, ka = args.ka,
                                leave_self_out = True, neighborhood_size = args.neighborhood_size)

    imputed_data = impute(feature_matrix = raw_data, diffusion_matrix = markov_matrix)

    print('Saving imputed matrix ...', file = sys.stderr)
    np.save(args.output_prefix+'.npy', imputed_data)

    if args.save_markov_matrix:
        print('Saving markov matrix ...', file = sys.stderr)
        sparse.save_npz(args.output_prefix+'_markov.npy',markov_matrix)