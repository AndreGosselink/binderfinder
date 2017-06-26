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

# def iterate_to(X, m, std, lower, upper):
#     Y = np.random.normal(m, std, X.size)
#     base = X.copy()
#     best = np.cov(base, Y)[0, 1]
#     tries = 0
#     diff = 0
#     misses = 0
#     while not lower < best < upper:
#         
#         if diff < 0.0001:
#             misses += 1
# 
#         if misses > 15000:
#             Y = np.random.normal(m, std, X.size)
#             best = np.cov(base, Y)[0, 1]
#             tries = 0
#             diff = 0
#             misses = 0
# 
#         tries += 1
#         i = np.random.randint(Y.size)
#         j = np.random.randint(Y.size)
# 
#         y = np.random.randint(Y.size)
#         Y[i], Y[j] = Y[j], Y[i]
# 
#         cur = np.cov(base, Y)[0, 1]
#         if best > upper:
#             better = cur < best
#         elif best < lower:
#             better = cur > best
#         if not better:
#             Y[i], Y[j] = Y[j], Y[i]
#             diff = 0
#             continue
#         diff = np.abs(best-cur)
#         best = cur
#         print '\r{: 5d}: {:.5f}({:.5f}|{:> 5d})'.format(tries, best, diff, misses),
#     print
#     return Y
# 

def iterate_data(params, base_a, cap1):
    base_idx = params.index('base')
    zero_idx = params.index('zero')
    pos_idx = params.index('pos')
    neg_idx = params.index('neg')
    
    x = np.linspace(-2, 2, n)

    sig = gauss(x, base_a, 0, 1) + np.random.normal(0, base_a*0.03, n)
    sig += gauss(x, base_a/3.0, 1, 0.2)
    sig += gauss(x, base_a/2.0, 0, 0.15)
    sig += 2*np.sin(x*np.pi)
    var_sig = np.var(sig, ddof=1)

    data = np.zeros((p, n), float)
    
    tries = 0
    zero_cap = np.inf
    for idx, bv in enumerate(sig):
        data[base_idx, idx] = bv
        zero_ok = pos_ok = neg_ok0 = neg_ok1 = False
        while 1:
            tries += 1
            if tries >= 5000:
                return iterate_data(params, base_a, cap1)
            print '\r{:> 3d} {:d} {:d} ({:d}, {:d}) ({:> 3.4f})'.format(idx, zero_ok, pos_ok, neg_ok0, neg_ok1, zero_cap),
            if not pos_ok:
                data[pos_idx, idx] = np.random.normal(bv, 1)
            if not zero_ok:
                data[zero_idx, idx] = np.random.normal(0, 3)
            if not neg_ok0 or not neg_ok1:
                data[neg_idx, idx] = np.random.normal(-bv, 1) 

            cov = np.cov(data)
            base_var = cov[base_idx,base_idx]

            dcap = -((base_var-cap1)/float(n-1))
            zero_cap = dcap * idx + base_var

            zero_ok = -zero_cap < cov[base_idx,zero_idx] < zero_cap
            pos_ok = cov[base_idx,pos_idx] >= 0.99 * base_var
            var_zn = cov[neg_idx,neg_idx] * zero_cap
            neg_ok0 = cov[base_idx,neg_idx] <= -0.99 * base_var
            neg_ok1 = -zero_cap < cov[neg_idx,zero_idx] < zero_cap

            if zero_ok and pos_ok and neg_ok0 and neg_ok1:
                break
    return data

def gauss(x, a, b, c):
    return a * np.exp(-(x-b)**2 / c**2)

with open('./data/mock_data_pca_rnd.csv', 'w') as df:
    n = 500
    params = ['base', 'zero', 'pos', 'neg']
    base_a = 5
    p = len(params)
    fmt = '{:.2f}'.format
    cap1 = 0.0001

    data = iterate_data(params, base_a, cap1)
    
    cov = np.cov(data)
    print
    print cov

    f, ax = plt.subplots(4,4)
    for i in xrange(4):
        for j in xrange(4):
            if i == 0:
                ax[i,j].set_title(params[j])
            if j == 0:
                ax[i,j].set_ylabel(params[i])

            ax[i,j].scatter(data[i], data[j], edgecolor='none')
            ax[i,j].set_xlabel('{:.5f}'.format(cov[i,j]))

    f.tight_layout()

    f2, ax = plt.subplots(2,2)
    ax[0,0].plot(data[0])
    ax[0,1].plot(data[1])
    ax[1,0].plot(data[2])
    ax[1,1].plot(data[3])

    data = data.T
    
    df.write('properties;0\n')
    df.write('parameters;{};{}\n'.format(len(params), ';'.join(params)))
    for i in xrange(n):
        df.write(';'.join(map(fmt, data[i])) + '\n')

    plt.show()
