import numpy as np

def dot(a, b):
    n = len(a)
    m = len(b)
    res = [[None for i in xrange(m)] for j in xrange(n)]
    for i in xrange(m):
        for j in xrange(n):
            val = 0
            for k in xrange(m):
                val += a[j][k] * b[k][i]
            res[j][i] = val
    return np.array(res)

a = np.arange(18).reshape(6, 3)
u = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], float)

m = np.array([[1, 0, 1],
              [0, 1, 0],
              [0, 0, 1]], float)

print a
print u
print np.dot(a, u)
print np.dot(a, m)
