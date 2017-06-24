import math
import numpy as np

def mean(X):
    i = 0.0
    s = 0.0
    for xi in X:
        s += xi
        i += 1.0
    return s / i

def var(X):
    mX = mean(X)
    i = 0.0
    s = 0.0
    for xi in X:
        s += (xi - mX)**2
        i += 1.0
    return s / (i - 1.0)
    
def std(X):
    vX = var(X)
    return math.sqrt(vX)

def cov(X, Y):
    if len(X) != len(Y):
        raise ValueError('Inputs need same Dimensions')
    mX = mean(X)
    mY = mean(Y)
    i = 0.0
    s = 0.0
    for xi, yi in zip(X, Y):
        s += (xi - mX) * (yi - mY)
        i += 1.0
    return s / (i - 1.0)

def covmat(*params):
    dims = len(params)

    matrix = []

    for i in xrange(dims):
        row = []
        for j in xrange(dims):
            row.append(cov(params[i], params[j]))
        matrix.append(row)

    return matrix


if __name__ == '__main__':
    X = [1, -1, 4]
    Y = [2, 1, 3]
    Z = [1, 3, -1]

    dat = np.column_stack([X, Y, Z]).T
    print dat

    print 'covs'
    print 'cov(X, X): {:.2f} {:.2f}'.format(cov(X, X), var(X))
    print 'cov(Y, Y): {:.2f} {:.2f}'.format(cov(Y, Y), var(Y))
    print 'cov(Z, Z): {:.2f} {:.2f}'.format(cov(Z, Z), var(Z))

    print 'mine'
    for r in covmat(X, Y, Z):
        print (' {:> 3.2f}'*len(r)).format(*r)

    print 'numpy'
    for r in np.cov(dat):
        print (' {:> 3.2f}'*len(r)).format(*r)

    'eigenvectors'
    print np.linalg.eig(np.cov(dat))
    for r in np.linalg.eig(np.cov(dat)):
        print r
        # print (' {:> 3.2f}'*len(r)).format(*r)
