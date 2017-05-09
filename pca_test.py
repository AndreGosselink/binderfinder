# http://sebastianraschka.com/Articles/len014_pca_step_by_step.html#using-the-pca-class-from-the-matplotlibmlab-library

# good documentation... fuck you, whales!
# https://www.clear.rice.edu/comp130/12spring/pca/pca_docs.shtml

import numpy as np
from matplotlib.mlab import PCA as mlabPCA
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

from binderfinder.pca import PCA as pca_bf
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(144337) 


def add_arows_mlab(ax, score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    # ax.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        tox = coeff[i, 0] / scalex
        toy = coeff[i, 1] / scaley
        # not a real projection!
        ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
        if labels is None:
            ax.text(tox * 1.15, toy * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

def add_arows_sklearn(ax, score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    # ax.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        tox = coeff[i, 0] / scalex
        toy = coeff[i, 1] / scaley
        ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
        if labels is None:
            ax.text(tox * 1.15, toy * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

def add_arows_bf(ax, score, coeff, labels=None):
    xs = score[0,:]
    ys = score[1,:]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    # ax.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        tox = coeff[i, 0]  / scalex
        toy = coeff[i, 1]  / scaley
        ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
        if labels is None:
            ax.text(tox * 1.15, toy * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

def biplot_bf(ax, score, coeff, labels=None):
    xs = score[0,:]
    ys = score[1,:]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    ax.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        tox = coeff[i, 0]#  / scalex
        toy = coeff[i, 1]#  / scaley
        ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
        if labels is None:
            ax.text(tox * 1.15, toy * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

def biplot_mlab(ax, score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    ax.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        tox = coeff[i, 0]
        toy = coeff[i, 1]
        ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
        if labels is None:
            ax.text(tox * 1.15, toy * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

def biplot_sklearn(ax, score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    ax.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        tox = coeff[i, 0]
        toy = coeff[i, 1]
        ax.arrow(0, 0, tox, toy, color='r', alpha=0.5)
        if labels is None:
            ax.text(tox * 1.15, toy * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            ax.text(tox * 1.15, toy * 1.15, labels[i], color='g', ha='center', va='center')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

len0 = 200
len1 = 400

# mock data
mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1000]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, len0).T

mu_vec2 = np.array([0, 0, 0])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1000]])
class2_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, len0).T

all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
print all_samples.T.shape, all_samples.shape

f, ax = plt.subplots(3, 4)

f3d = plt.figure()
ax3d = [f3d.add_subplot(1, 2, i, projection='3d') for i in xrange(1, 3)]

# matlab geroedel
mlab_pca = mlabPCA(all_samples.T)
ax[0, 0].plot(mlab_pca.Y[0:len0,0],mlab_pca.Y[0:len0,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
ax[0, 0].plot(mlab_pca.Y[len0:len1,0], mlab_pca.Y[len0:len1,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
ax3d[0].scatter(*[mlab_pca.Y[:len0,i] for i in xrange(3)], s=7, c='blue', alpha=0.5)
ax3d[0].scatter(*[mlab_pca.Y[len0:,i] for i in xrange(3)], s=7, c='red', alpha=0.5)

ax[0, 0].set_xlabel('x_values')
ax[0, 0].set_ylabel('y_values')
ax[0, 0].set_xlim([-4,4])
ax[0, 0].set_ylim([-4,4])
ax[0, 0].set_title('mlab')
plt.legend()

ax[1,0].scatter(xrange(3), mlab_pca.fracs)
plt.legend()


# sklearn
stdScaler = StandardScaler()
sklearn_pca = sklearnPCA(n_components=3, svd_solver='auto')
# sklearn_transf = sklearn_pca.fit_transform(stdScaler.fit_transform(np.flipud(np.fliplr(all_samples.T))))
sklearn_transf = sklearn_pca.fit_transform(stdScaler.fit_transform(all_samples.T))

ax[0, 1].plot(sklearn_transf[0:len0,0],sklearn_transf[0:len0,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
ax[0, 1].plot(sklearn_transf[len0:len1,0], sklearn_transf[len0:len1,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
ax3d[1].scatter(*[sklearn_transf[:len0,i] for i in xrange(3)], s=7, c='blue', alpha=0.5)
ax3d[1].scatter(*[sklearn_transf[len0:,i] for i in xrange(3)], s=7, c='red', alpha=0.5)

ax[0, 1].set_xlabel('x_values')
ax[0, 1].set_ylabel('y_values')
ax[0, 1].set_xlim([-4,4])
ax[0, 1].set_ylim([-4,4])
ax[0, 1].set_title('sklearn')
plt.legend()

ax[1, 1].scatter(xrange(len(sklearn_pca.explained_variance_ratio_)), sklearn_pca.explained_variance_ratio_)

# own implementation cov
bf_pca = pca_bf(all_samples.T)

ax[0, 2].plot(bf_pca.cov_transform[0,:len0], bf_pca.cov_transform[1,:len0], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
ax[0, 2].plot(bf_pca.cov_transform[0,len0:], bf_pca.cov_transform[1,len0:], '^', markersize=7, color='red', alpha=0.5, label='class2')

ax[0, 2].set_xlabel('x_values')
ax[0, 2].set_ylabel('y_values')
ax[0, 2].set_title('own_cov')
plt.legend()

ax[0, 3].plot(bf_pca.cor_transform[0,:len0], bf_pca.cor_transform[1,:len0], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
ax[0, 3].plot(bf_pca.cor_transform[0,len0:], bf_pca.cor_transform[1,len0:], '^', markersize=7, color='red', alpha=0.5, label='class2')

ax[0, 3].set_xlabel('x_values')
ax[0, 3].set_ylabel('y_values')
ax[0, 3].set_title('own_cor')
plt.legend()

ax[1, 2].scatter(xrange(len(bf_pca.eig_val_cov)), bf_pca.eig_val_cov)
ax[1, 3].scatter(xrange(len(bf_pca.eig_val_cor)), bf_pca.eig_val_cor)

 
add_arows_mlab(ax[0, 0], mlab_pca.Y[:,0:2], mlab_pca.Wt[:,0:2])
biplot_mlab(ax[2, 0], mlab_pca.Y[:,0:2], mlab_pca.Wt[:,0:2])

add_arows_sklearn(ax[0, 1], sklearn_transf, sklearn_pca.components_)
biplot_sklearn(ax[2, 1], sklearn_transf, sklearn_pca.components_)
eigs = sklearn_pca.components_ * 4
for i in xrange(3):
    ax3d[1].plot(*[[0, eigs[i][j]] for j in xrange(3)], lw=3)
      

add_arows_bf(ax[0, 2], bf_pca.cov_transform, bf_pca.eig_vec_cov)
biplot_bf(ax[2, 2], bf_pca.cov_transform, bf_pca.eig_vec_cov)

add_arows_bf(ax[0, 3], bf_pca.cor_transform, bf_pca.eig_vec_cor)
biplot_bf(ax[2, 3], bf_pca.cor_transform, bf_pca.eig_vec_cor)

plt.show()
