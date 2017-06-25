"""
Implemented according to:

'A tutorial on Principal Components Analysis', Lindsay I Smith, 2002
"""



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
                 normalize=True,
                 reduce_to=-1,
                 figsize=(6, 11),
                 debug=False,
                ):

        offset_func = {  'mean': lambda dat: np.mean(dat, axis=1),
                       'median': lambda dat: np.median(dat, axis=1),
                      }

        self.debug = debug
        self._normalize = normalize
        self._figsize = figsize
        self._offset_func = offset_func[centering]

        self.parser = parser = Parser(filename)
        self._filename = filename
        
        # parameter in columns, items in rows
        _, _, data = parser.get_pca_formatted()
        #!---> parameter in rows, items in columns!
        self.data = data.T

        if reduce_to == -1:
            self._reduce = self.data.shape[0]
        elif reduce_to > self.data.shape[0]:
            raise ValueError('trying data reduction to more dimensions than in data space!')
        else:
            self._reduce = reduce_to

        # standardize
        self.data = self._standardize_data(self.data)

        self._covMat = np.cov(self.data)
        self._eigenvals, self._eigenvecs = self._get_eigenpairs(self._covMat)
        self._sortidx = sortidx = np.argsort(self._eigenvals)[::-1]
        # switch rows/cols
        #TODO look at this... and....
        self._fVecs = self._eigenvecs[sortidx][:self._reduce].squeeze()
        self._fVals = self._eigenvals[sortidx][:self._reduce].squeeze()

        self._check_consistency()

        self.pca_transform = self._transform(self.data)

        # nornmalizing
        self.pca_transform /= np.max(self.pca_transform)
        
        if debug and self.data.shape[0] == 2:
            print 'data'
            print self.data
            print 'covMat'
            print self._covMat
            
            print 'eigenpairs'
            for n, (val, vec) in enumerate(zip(self._eigenvals[sortidx],
                                              self._eigenvecs[sortidx])):
                print 'PC{}'.format(n+1), val, vec

            print 'featureVector'
            print self._fVecs

    def _transform(self, data):
        # rowx by row!
        #TODO this here
        pca_transform = np.dot(self._fVecs.T, data)
        return pca_transform

    def _get_eigenpairs(self, False):
        m = self._covMat.shape[0]
        evals, evecs = np.linalg.eig(self._covMat)
        # for dup in eigenpairs:
        #     print '{:> 3.3f} -> ({:})'.format(dup[0], ', '.join(map('{:> 3.3f}'.format, dup[1])))
        # 

        return evals, evecs.T

    def _standardize_data(self, data):
        if self._normalize:
            data = data.copy() 
            for i, p in enumerate(data):
                data[i] /= np.max(p)

        m = data.shape[0]
        offsets = self._offset_func(data)
        # for broadcasting reasons
        offsets = offsets.reshape(m, 1)
        res = data - offsets
        return res

    def _check_consistency(self):
        print 'checking eigenvector properties...',
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
        print 'all good!'
    
    def show(self):
        f, ax = plt.subplots(2, figsize=self._figsize)
        ax[0].scatter(*self.pca_transform[:2,:])
        ax[0].set_xlim(-1.3, 1.3)
        ax[0].set_ylim(-1.3, 1.3)
        ax[0].set_xlabel('PC 1')
        ax[0].set_ylabel('PC 2')
        ax[0].set_title(self._filename + ' PCA')

        self._draw_arrows(ax[0])

        sortidx = self._sortidx
        pcnum = range(self._fVals.size)
        ax[1].scatter(pcnum, self._fVals)
        ax[1].set_xticks(pcnum)
        ax[1].set_xticklabels(['PC{}'.format(i+1) for i in pcnum])
        ax[1].set_title('Proportion of Variance')
        
        f.tight_layout()
        plt.show()

    def _draw_arrows(self, ax):
        n = self._reduce
        # labels = [self.parser.data_layout[self.parser.PARA_LABEL][idx] for idx in self._sortidx]
        labels = self.parser.data_layout[self.parser.PARA_LABEL]
        unit_vecs = np.zeros((n, n), float)
        for i in xrange(n):
            unit_vecs[i, i] = 1
        biplots = self._transform(unit_vecs.T)
        for i in range(n):
            # get vectors
            tox, toy = biplots[:2, i]
            ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

        
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
            ax.text(tox * 1.1, toy * 1.1, labels[i], color='g', ha='center', va='center')

        ax.set_xlabel("PC{}".format(1))
        ax.set_ylabel("PC{}".format(2))
        ax.set_title('Biplot')
