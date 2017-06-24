import numpy as np
from matplotlib.mlab import PCA as mlabPCA
import matplotlib.pyplot as plt
from dataparser import Parser
from errors import DeprecatedDependency
import warnings
import os
from misc import covmat


class PCA(object):
    """
Input:
------------------------
 filename: path to csv file, containing parsable data

 standardize: True if input data are to be standardized. If False, only centering will be carried out.
 """
    

    def __init__(self, filename,
                 centering='mean',
                 reduce_to=-1,
                 figsize=(6, 11),
                 debug=False,
                ):

        offset_func = {  'mean': lambda dat: np.mean(dat, axis=1),
                       'median': lambda dat: np.median(dat, axis=1),
                      }

        self.debug = debug
        self._offset_func = offset_func[centering]

        parser = Parser(filename)
        
        # parameter in columns, items in rows
        _, _, data = parser.get_pca_formatted()
        #!---> parameter in rows, items in columns!
        self.data = data.T

        if reduce_to == -1:
            self._reduce = self.data.shape[0]
        elif reduce_to > self.data.shape[0]:
            raise ValueError('trying data reduction to more dimensions than in data space!')
        
        # standardize
        self.data = self._standardize_data(self.data)


        # print subset
        # for l, p in zip(parser.data_layout[parser.PARA_LABEL], subset):
        #     print l, p, np.var(p, ddof=1)
        # print 'mine'
        # for r in covmat(*subset):
        #     print (' {:> 3.5f}'*len(r)).format(*r)
        # print 'numpy'
        # for r in covMat:
        #     print (' {:> 3.5f}'*len(r)).format(*r)

        self._covMat = np.cov(self.data)
        self._eigenvals, self._eigenvecs = self._get_eigenpairs(self._covMat)
        self._sortidx = np.argsort(self._eigenvals)

        self._check_consistency()
        
        if debug and self.data.shape[0] == 2:
            print 'data'
            print self.data
            print 'covMat'
            print self._covMat
            
            scale_to = np.max(self._eigenvals)
            plt.scatter(self.data[0], self.data[1])
            for val, vec in zip(self._eigenvals, self._eigenvecs):
                plt.plot([0, vec[0] * val / scale_to], [0, vec[1] * val / scale_to])
            print 'eigenpairs'
            sortidx = self._sortidx[::-1]
            for n, (val, vec) in enumerate(zip(self._eigenvals[sortidx],
                                              self._eigenvecs[sortidx])):
                print 'PC{}'.format(n+1), val, vec

            plt.show()

    def _get_eigenpairs(self, covMat):
        m = covMat.shape[0]
        evals, evecs = np.linalg.eig(covMat)
        # for dup in eigenpairs:
        #     print '{:> 3.3f} -> ({:})'.format(dup[0], ', '.join(map('{:> 3.3f}'.format, dup[1])))
        # 

        return evals, evecs.T

    def _standardize_data(self, data):
        m = data.shape[0]
        offsets = self._offset_func(data)
        # for broadcasting reasons
        offsets = offsets.reshape(m, 1)
        res = data - offsets
        return res

    def _check_consistency(self):
        print 'checking eigenvector properties...'
        m = self._covMat.shape[0]
        vecs = self._eigenvecs
        vals = self._eigenvals
        for i in xrange(m):
            is_close = np.all(np.isclose((np.dot(self._covMat,
                vecs[i]) / vals[i]), vecs[i]))
            if not is_close:
                raise ValueError('Eigenvector {} is odd...'.format(i))
            if not np.isclose(np.linalg.norm(vecs[i]), 1.0):
                raise ValueError('Eigenvector {} is not normed...'.format(i))
        print 'all good'
        
class Mlab_PCA(object):
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

        # n = mlab_pca.Wt.shape[0]
        unit_vectors = np.zeros((n, n), float)
        for i in xrange(n):
            unit_vectors[i, i] = 1

        print unit_vectors
        unit_vectors = mlab_pca.center(unit_vectors)
        print unit_vectors

        bivectors = mlab_pca.project(unit_vectors)

        max_val = np.max([xs.max(), ys.max()])
        max_to = np.max(np.sqrt(np.sum(bivectors**2, axis=0)))
        ax.scatter(xs/max_val, ys/max_val)
        for i in range(n):
            # get vectors
            # v = np.zeros(mlab_pca.Wt.shape[1])
            # v[i] = 1
            tox, toy = bivectors[i, :2] / max_to
            # bp = mlab_pca.project(v)

            # tox = bp[0] / max_to
            # toy = bp[1] / max_to
            # not a real projection!
            ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

        ax.set_xlabel("PC{}".format(1))
        ax.set_ylabel("PC{}".format(2))
        ax.set_title('Biplot')
