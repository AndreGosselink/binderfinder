"""
Implemented according to:

'A tutorial on Principal Components Analysis', Lindsay I Smith, 2002

and with ideas from:
Teddy Roland (https://github.com/teddyroland/python-biplot/blob/master/biplot.py)
"""

import numpy as np
from matplotlib.mlab import PCA as mlabPCA
import matplotlib.pyplot as plt
from dataparser import Parser
from errors import DeprecatedDependency
import warnings
import os
from misc import covmat
from . import __version__


class PCA(object):
    """
Input:
------------------------
    filename: path to csv file, containing parsable data

    centering: centering data by 'mean', 'median' or 'none' centering at all

    normalize: normalize each parameter independently prior PCA

    reduce_to: number of components the data is reduced to. If -1 all components are used

    annotate: annotate datapoint in scatterplot. where 0 is the first data item in the data file

    portions: plot portion of variance for each component
 """
    

    def __init__(self, filename,
                 centering='mean',
                 normalize=False,
                 reduce_to=-1,
                 figsize=(),
                 debug=False,
                 portions=True,
                 annotate=False,
                 covplot=False,
                ):

        offset_func = {  'mean': lambda dat: np.mean(dat, axis=1),
                       'median': lambda dat: np.median(dat, axis=1),
                         'none': lambda dat: dat,
                      }

        self.debug = debug
        self._normalize = normalize
        self._offset_func = offset_func[centering]
        self._portions = portions
        self._annotate = annotate

        self._covplot = covplot

        if figsize == () and portions:
            self._figsize = (13, 7)
        else:
            self._figsize = (7, 6.5)

        self.parser = parser = Parser(filename)
        self._filename = filename
        
        # parameter in columns, items in rows
        _, _, data = parser.get_pca_formatted()
        self._labels = np.array(parser.data_layout[parser.PARA_LABEL])
        #!---> parameter in rows, items in columns!
        self.data = data.T

        # sorting dat by variance
        variances = np.var(self.data, ddof=1, axis=1)
        sorter = np.argsort(variances)
        self.data = self.data[sorter]
        self._labels = self._labels[sorter]

        if reduce_to == -1:
            self._reduce = self.data.shape[0]
        elif reduce_to > self.data.shape[0]:
            raise ValueError('trying data reduction to more dimensions than available in data space!')
        else:
            self._reduce = reduce_to

        # standardize
        self.data = self._standardize_data(self.data)

        self._covMat = np.cov(self.data)
        self._eigenvals, self._eigenvecs = self._get_eigenpairs(self._covMat)
        self._sortidx = sortidx = np.argsort(self._eigenvals)[::-1]
        # switch rows/cols
        #TODO look at this... and....
        self._fVecs = self._eigenvecs[sortidx][:self._reduce].T.squeeze()
        self._fVals = self._eigenvals[sortidx][:self._reduce].T.squeeze()

        self.pca_transform = self._transform(self.data)

        # # nornmalizing
        # self.pca_transform /= np.max(self.pca_transform)
        
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

        self._check_consistency()

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
            if self.debug:
                print 'normalizing data...'
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
        if self.debug:
            print 'checking eigenvector properties...',
        m = self._covMat.shape[0]
        vecs = self._eigenvecs
        vals = self._eigenvals
        for i in xrange(m):
            is_close = np.all(np.isclose((np.dot(self._covMat,
                vecs[i]) / vals[i]), vecs[i]))
            if not is_close:
                print '\n\ndim\tis\t\tought\t\tdiff\t\tpass'
                for i, (v, r) in enumerate(zip(np.dot(self._covMat, vecs[i]) / vals[i], vecs[i])):
                    print '{:d}\t{: 2.3e}\t{: 2.3e}\t{: 2.3e}\t{:}'.format(i, v, r, v-r, np.isclose(v, r))
                raise ValueError('Eigenvector {} is odd, Try normalize=True'.format(i))
            if not np.isclose(np.linalg.norm(vecs[i]), 1.0):
                raise ValueError('Eigenvector {} is not normed...'.format(i))
        if self.debug:
            print 'all good!'
    
    def show(self):

        xmin, xmax = map(lambda f: f(self.pca_transform[0,:])*1.3, [np.min, np.max])
        ymin, ymax = map(lambda f: f(self.pca_transform[1,:])*1.3, [np.min, np.max])

        absmin = np.min([xmin, ymin])
        absmax = np.max([xmax, ymax])
        scale_to = np.min([np.abs(absmin), absmax]) * 0.8

        sortidx = self._sortidx
        pcnum = range(self._fVals.size)

        if self._portions:
            f, ax = plt.subplots(1, 2, figsize=self._figsize)
            ax[0].scatter(*self.pca_transform[:2,:])
            ax[0].set_xlim(absmin, absmax)
            ax[0].set_ylim(absmin, absmax)
            ax[0].set_xlabel('PC 1')
            ax[0].set_ylabel('PC 2')
            ax[0].set_title('PCA/BiPlot')
            arrow_ax = ax[0]

            ax[1].scatter(pcnum, self._fVals / np.sum(self._fVals))
            ax[1].set_xticks(pcnum)
            ax[1].set_xticklabels(['PC{}'.format(i+1) for i in pcnum])
            ax[1].set_title('Proportion of Variance')
        else:
            f, ax = plt.subplots(1, figsize=self._figsize)
            ax.scatter(*self.pca_transform[:2,:])
            ax.set_xlim(absmin, absmax)
            ax.set_ylim(absmin, absmax)
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title(self._filename + ' PCA')
            arrow_ax = ax

        self._draw_arrows(arrow_ax, scale_to)
        f.canvas.set_window_title('binderfinder PCA {} -- {}'.format(__version__, self._filename))
        f.tight_layout()

        if self._annotate:
            self._annotate_plot(arrow_ax)

        if self._covplot:
            n = self.data.shape[0]
            fcp, ax = plt.subplots(n, n, sharex=True, sharey=True)
            fcp.canvas.set_window_title('binderfinder CovPlot {} -- {}'.format(__version__, self._filename))
            fcp.subplots_adjust(hspace=0, wspace=0)
            for i in xrange(n):
                for j in xrange(n):
                    if i == 0:
                        ax[i, j].set_title(self._labels[j])
                    if j == 0:
                        ax[i, j].set_ylabel(self._labels[i])
                    ax[i, j].scatter(self.data[j], self.data[i], edgecolor='none', s=1)



        plt.show()

    def _draw_arrows(self, ax, scale_to):
        n = self._reduce
        # labels = [self.parser.data_layout[self.parser.PARA_LABEL][idx] for idx in self._sortidx]
        labels = self._labels
        if self.debug:
            print 'Picking Arrows and drawing them'
        bivecs = self._fVecs[:,:2]
        length = np.max(np.sqrt((self._fVecs[:,0] + self._fVecs[:,1])**2))
        if self.debug:
            for i in xrange(2):
                val = self._fVals[i]
                vec = bivecs[:,i]
                tvec = np.dot(self._covMat, vec) / val
                is_close = np.all(np.isclose(tvec, vec))
                if self.debug:
                    print is_close,
                    if not is_close:
                        print '({}) -> ({})'.format(','.join(map('{:.2f}'.format, tvec)), ','.join(map('{:.2f}'.format, vec)))
        for i in range(n):
            # get vectors
            tox = bivecs[i,0] / length * scale_to
            toy = bivecs[i,1] / length * scale_to
            ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

    def _annotate_plot(self, ax):
        pc0 = self.pca_transform[0,:]
        pc1 = self.pca_transform[1,:]
        offset = np.max([pc0, pc1])*0.03
        for idx, (x, y) in enumerate(zip(pc0, pc1)):
            ax.text(x, y+offset, str(idx), color=[0.0, 0.0, 0.0, 0.5], ha='center', va='center', size='xx-small')
