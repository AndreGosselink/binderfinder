import random as rnd
import numpy as np
import matplotlib.pyplot as plt

# with open('./data/mock_data.csv', 'w') as df:
#     df.write('properties;2\n')
#     df.write('parameters;2\n')
#     for i, t in enumerate('ABCDEFGHIJ'):
#         for j, s in enumerate('abcde'):
#             df.write('{};{};{};{}\n'.format(t, s, i*25+rnd.random()*25, j*2.5+rnd.random()*2.5))
# 
# with open('./data/mock_data_rnd.csv', 'w') as df:
#     df.write('properties;2\n')
#     df.write('parameters;2\n')
#     for i, t in enumerate('ABCDEFGHIJ'):
#         for j, s in enumerate('abcde'):
#             df.write('{};{};{};{}\n'.format(t, s, rnd.random()*25, rnd.random()*2.5))
# 
# with open('./data/mock_data_ones.csv', 'w') as df:
#     df.write('properties;2\n')
#     df.write('parameters;2\n')
#     for i, t in enumerate('ABCDEFGHIJ'):
#         for j, s in enumerate('abcde'):
#             df.write('{};{};{};{}\n'.format(t, s, 2.0, rnd.random()))
# 
# with open('./data/mock_data_oneszeros.csv', 'w') as df:
#     df.write('properties;2\n')
#     df.write('parameters;2\n')
#     for i, t in enumerate('ABCDEFGHIJ'):
#         for j, s in enumerate('abcde'):
#             df.write('{};{};{};{}\n'.format(t, s, 1, 0))

# with open('./data/mock_data_pca_shapes.csv', 'w') as df:
#     n = 500
#     p = 5
#     fmt = '{:.2f}'.format
#     all_high = 0
#     data = np.zeros((n, p))
# 
#     a = np.random.uniform(0, 5, n)
#     b = np.random.uniform(0, 5, n)
# 
#     circles = np.abs(a - b) <= a * 0.85
#     ellipses = np.logical_and(0.49 * a <= b, b <= 0.86 * a)
#     rectangles = np.logical_and(~circles, ~ellipses)
# 
#     print np.sum(circles)
#     print np.sum(ellipses)
#     print np.sum(rectangles)
# 
#     diag = np.sqrt(a**2 + b**2)
#     
#     # rectangles
#     data[:,0] = a
#     data[:,1] = b
#     data[:,2] = diag
#     data[:,3] = 2*a + 2*b
#     data[:,4] = a * b
#     
#     # circle
#     data[:,0] = a
#     data[:,1][circles] = a[circles]
#     data[:,2][circles] = a[circles]
#     data[:,3][circles] = np.pi * a[circles]
#     data[:,4][circles] = np.pi * (a[circles]/2)**2
#     
#     # ellipsis
#     t = np.pi / 4
#     cost = np.cos(t)
#     sint = np.sin(t)
#     x = a * cost
#     y = b * sint
#     data[:,1][ellipses] = 0.6938 * a[ellipses]
#     data[:,2][ellipses] = 2 * np.sqrt(x[ellipses]**2 + y[ellipses]**2)
#     data[:,3][ellipses] = 2 * np.pi * np.sqrt(0.5 * (a[ellipses]**2 + b[ellipses]**2))
#     data[:,4][ellipses] = np.pi * a[ellipses] * b[ellipses]
# 
# 
#     df.write('properties;0\n')
#     df.write('parameters;{};major;minor;diag;len;surf\n'.format(p))
#     for i in xrange(n):
#         df.write(';'.join(map(fmt, data[i])) + '\n')

# c = np.zeros((n,3))
# c[:,0][rectangles] = 1
# c[:,1][circles] = 1
# c[:,2][ellipses] = 1
# f, ax = plt.subplots(2)
# ax[0].scatter(data[:,1], data[:,2], edgecolor='none', s=3, c=c)
# plt.show()

def cov(X, Y):
    if len(X) != len(Y):
        raise ValueError('Inputs need same Dimensions')
    mX = np.mean(X)
    mY = np.mean(Y)
    i = 0.0
    s = 0.0
    for xi, yi in zip(X, Y):
        s += (xi - mX) * (yi - mY)
        i += 1.0
    return s / (i - 1.0)

def iterate_to(X, m, std, lower, upper):
    Y = np.random.normal(m, std, X.size)
    base = X.copy()
    best = np.cov(base, Y)[0, 1]
    misses = 0
    max_tries = 10
    tries = 0
    while not lower < best < upper:
        if misses > max_tries:
            Y = np.random.normal(m, std, X.size)
        tries += 1
        i = np.random.randint(Y.size)
        j = np.random.randint(Y.size)

        y = np.random.randint(Y.size)
        Y[i], Y[j] = Y[j], Y[i]

        cur = np.cov(base, Y)[0, 1]
        if best > upper:
            better = cur < best
        elif best < lower:
            better = cur > best
        if not better:
            Y[i], Y[j] = Y[j], Y[i]
            continue
            misses += 1
        best = cur
        print '\r{: 5d}({: 5d}): {:.5f}'.format(tries, misses, best),
    print
    return Y

with open('./data/mock_data_pca_rnd.csv', 'w') as df:
    n = 500
    params = ['base', 'zero', 'neg', 'pos']
    p = len(params)
    fmt = '{:.2f}'.format
    data = np.zeros((n, p), float)

    base = np.random.normal(0, 1, n)
    zero = iterate_to(base, 0, 1, 0.0, 0.0001)
    pos = iterate_to(base, 0, 1, 0.97, 100)
    neg = iterate_to(base, 0, 1, -100, -0.97)

    data[:,0] = base
    data[:,1] = zero
    data[:,2] = pos
    data[:,3] = neg

    df.write('properties;0\n')
    df.write('parameters;{};{}\n'.format(len(params), ';'.join(params)))
    for i in xrange(n):
        df.write(';'.join(map(fmt, data[i])) + '\n')
