import numpy as np


class PCA(object):

    def __init__(self, data_input):
        """
        NxM matrix, where N is the datapoint and M is the number of parameters
        """
        mean_vector = self.get_meanvector(data_input)
        cov_matrix = self.get_cov_matrix(data_input)

    def get_meanvector(self, data):
        return np.mean(data, axis=0)

    def get_cov_matrix(self, data):
        cov_mat = np.cov([data[:,0], data[:,1], data[:,2]])
        print cov_mat

        
