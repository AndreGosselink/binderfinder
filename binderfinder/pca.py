import numpy as np


class PCA(object):

    def __init__(self, data_input, red_dim=2):
        """
        NxM matrix, where N is the datapoint and M is the number of parameters
        """
        # for scatter matrix (whatevert this is)
        # mean_vector = self.get_meanvector(data_input)

        # matrices for eigenvector calculation
        cov_mat = self.get_cov_matrix(data_input)
        cor_mat = self.get_cor_matrix(data_input)
        # eigenvectors and values
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
        eig_val_cor, eig_vec_cor = np.linalg.eig(cor_mat)
        # sorting by eigenvalues
        sort_idx_cov = np.argsort(eig_val_cov)[::-1]
        sort_idx_cor = np.argsort(eig_val_cor)[::-1]
        eig_val_cov = eig_val_cov[sort_idx_cov]
        eig_val_cor = eig_val_cor[sort_idx_cor]
        eig_vec_cov = eig_vec_cov[sort_idx_cov]
        eig_vec_cor = eig_vec_cor[sort_idx_cor]
        # make transformation matrices
        trans_mat_cov = np.column_stack(eig_vec_cov[:red_dim])
        trans_mat_cor = np.column_stack(eig_vec_cor[:red_dim])
        # transform data onto subspace
        self.cov_transform = trans_mat_cov.T.dot(data_input.T)
        self.cor_transform = trans_mat_cor.T.dot(data_input.T)

        self.eig_vec_cov = eig_vec_cov
        self.eig_vec_cor = eig_vec_cor

        self.eig_val_cov = eig_val_cov
        self.eig_val_cor = eig_val_cor

    # def get_meanvector(self, data):
    #     return np.mean(data, axis=0)

    def get_cov_matrix(self, data):
        cov_mat = np.cov([data[:,0], data[:,1], data[:,2]])
        return cov_mat

    def get_cor_matrix(self, data):
        cor_mat = np.corrcoef([data[:,0], data[:,1], data[:,2]])
        return cor_mat

        
