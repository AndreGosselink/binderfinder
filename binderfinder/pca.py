import numpy as np
from matplotlib.mlab import PCA as mlabPCA
import matplotlib.pyplot as plt
from dataparser import Parser
from errors import DeprecatedDependency
import warnings
import os


class PCA(object):
    """
Input:
------------------------
 filename: path to csv file, containing parsable data

 standardize: True if input data are to be standardized. If False, only centering will be carried out.
 """

    def __init__(self, filename, standardize=True, figsize=(6, 11)):

        parser = Parser(filename)

        _, _, data = parser.get_pca_formatted()
        
        try:
            mlab_pca = mlabPCA(data, standardize)
        except Exception as e:
            print e
            warnings.warn('Please consider updating matplotlib. Unexpected call signatur for PCA', DeprecatedDependency)                    
            mlab_pca = mlabPCA(data, standardize)

        f, ax = plt.subplots(3, figsize=figsize)
        ax[0].plot(mlab_pca.Y[:,0], mlab_pca.Y[:,1], 'o', markersize=7, alpha=0.5)
        #
        ax[0].set_xlabel('PC 1')
        ax[0].set_ylabel('PC 2')
        ax[0].set_title(filename + ' PCA')
        
        pcnum = range(len(mlab_pca.fracs))
        ax[1].scatter(pcnum, mlab_pca.fracs)
        ax[1].set_xticks(pcnum)
        ax[1].set_xticklabels(['PC{}'.format(i+1) for i in pcnum])
        ax[1].set_title('Proportion of Variance')

        self.biplot_mlab(ax[2], mlab_pca.Y[:,0:2], mlab_pca, labels=parser.data_layout[parser.PARA_LABEL])
        
        f.tight_layout()
        plt.show()

    def biplot_mlab(self, ax, score, mlab_pca, labels):
         xs = score[:,0]
         ys = score[:,1]
         coeff = mlab_pca.Wt
         n = coeff.shape[0]
#         n = mlab_pca.Wt.shape[0]
         max_val = np.max([xs.max(), ys.max()])
         max_to = np.max(np.sqrt(mlab_pca.Wt[:,0]**2 + mlab_pca.Wt[:,1]**2))
         ax.scatter(xs/max_val, ys/max_val)
         for i in range(n):
             # get vectors
#             v = np.zeros(mlab_pca.Wt.shape[1])
#             v[i] = 1
             tox = coeff[i, 0] / max_to
             toy = coeff[i, 1] / max_to
#             bp = mlab_pca.project(v)

#             tox = bp[0] / max_to
#             toy = bp[1] / max_to
             # not a real projection!
             ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
             ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')
    
         ax.set_xlabel("PC{}".format(1))
         ax.set_ylabel("PC{}".format(2))
         ax.set_title('Biplot')
