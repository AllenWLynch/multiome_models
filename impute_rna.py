import anndata
import scanpy as sc
import numpy as np
import argparse
import breakpointer
import sys
import fire
from scipy import sparse
import time

def calculate_deviance(y_ij):

    y_ij = y_ij.todense()

    pi_j_hat = y_ij.sum(axis = 0) / y_ij.sum()

    n_i = y_ij.sum(axis = 1)

    mu_ij_hat = n_i * pi_j_hat

    count_dif = n_i - y_ij
    expected_count_dif = n_i - mu_ij_hat

    r_ij = np.multiply(
        np.sign(y_ij - mu_ij_hat), 
        np.sqrt(
          np.where(y_ij > 0, 2 * np.multiply(y_ij, np.log(y_ij / mu_ij_hat)), 0) + \
          2 * np.multiply(count_dif, np.log(count_dif / expected_count_dif))
        )
    )
    #assert( not np.isnan(np.where(y_ij > 0, 2*y_ij*np.log(y_ij / mu_ij_hat), 0)).any())
    return np.nan_to_num(r_ij)
  
def get_gene_deviance(y_ij):

    y_ij = y_ij.todense()

    pi_j_hat = y_ij.sum(axis = 0) / y_ij.sum()

    n_i = y_ij.sum(axis = 1)

    mu_ij_hat = n_i * pi_j_hat

    deviance_ij = np.where(y_ij > 0, np.multiply(y_ij, np.log(y_ij / mu_ij_hat)), 0) + np.multiply(n_i - y_ij, np.log((n_i - y_ij) / (n_i - mu_ij_hat)))

    deviance_j = 2*deviance_ij.sum(axis = 0)

    return deviance_j

def mark_highly_deviant_genes(andata, top_n = 1500):
    
    gene_deviances = get_gene_deviance(andata.X)
    andata.var['gene_deviance'] = np.array(gene_deviances).reshape(-1)
    andata.var['deviance_rank'] = andata.var.gene_deviance.rank(ascending=False)
    andata.var['highly_variable'] = andata.var.deviance_rank <= top_n

def sparse_dot_dense(sparse_matrix, dense_matrix):
    sparse_matrix = sparse_matrix.tocsr()

    result = []
    for i in range(sparse_matrix.shape[0]):
        nonzero_mask = sparse_matrix[i, :].indices
        nonzero_dense_elements = dense_matrix[nonzero_mask, :]

        result.append(
            np.dot(sparse_matrix[i, :].data, nonzero_dense_elements)
        )
        print("\rCompleted row {}".format(str(i+1)), file = sys.stderr, end = '')

    return np.vstack(result)

def sparse_test(m,n,o, density = 0.3):

    print('Formatting matrices ...')
    sparse_matrix = sparse.random(m,n,density = 0.3).tocsr()
    dense_matrix = np.random.randn(m, o)
    
    print("Starting tests ...")
    start = time.time()
    a1 = sparse_dot_dense(sparse_matrix, dense_matrix)
    end = time.time()
    print('Time elapsed: ' + str(end - start))

    start = time.time()
    a2 = sparse_matrix.dot(dense)
    end = time.time()
    print('Time elapsed: ' + str(end - start))

    print('Is close: ' + str(np.isclose(a1, a2)))


def main(data_path, save_path, min_cells = 10, min_genes = 200, num_variable_genes = 1500, ka = 4, neighborhood_size = 4,
    diffusion_time = 4):

    print('Reading ' + data_path + '...', file = sys.stderr)
    rna_data = anndata.read_h5ad(data_path)

    print('Cleaning data matrix ...', file = sys.stderr)
    rna_data = rna_data[:, ~rna_data.var_names.str.startswith('mt-')]
    sc.pp.filter_genes(rna_data, min_cells=min_cells)
    sc.pp.filter_cells(rna_data, min_genes=min_genes)
    
    print('Marking highly deviant genes ...', file = sys.stderr)
    mark_highly_deviant_genes(rna_data)

    print('Calculating residuals ...', file = sys.stderr)
    deviance_resis = calculate_deviance(rna_data.X)

    rna_data.raw = rna_data

    print('Performing preliminary PCA ...', file = sys.stderr)
    rna_data.X = np.array(deviance_resis)
    sc.pp.scale(rna_data, max_value=10)

    sc.tl.pca(rna_data, use_highly_variable=True)
    rna_data.obsm['rna_pca'] = rna_data.obsm['X_pca'][:,1:]

    print('Calculating Markov matrix ...', file = sys.stderr)
    markov_matrix = breakpointer.calc_markov_diffusion_matrix(
                    rna_data.obsm['rna_pca'],
                    ka = ka, neighborhood_size= neighborhood_size, diffusion_time=diffusion_time
                )

    print('Imputing gene expression matrix ...', file = sys.stderr)
    rna_data.layers['imputed'] = breakpointer.impute(
                rna_data.X, 
                markov_matrix
            )

    print('Saving data ...', file = sys.stderr)
    rna_data.write_h5ad(save_path)

    print('Done!', file = sys.stderr)

if __name__ == "__main__":
    fire.Fire(main)

    