import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    half_idx = params.index('half')
    feat_idx = params.index('feat')
    inter_idx = params.index('inter')
    
    x = np.linspace(-2, 2, n)

    sig = gauss(x, base_a, -1, 1.5) + np.random.normal(0, base_a*0.03, n)
    sig += gauss(x, base_a/3.0, 1, 0.2)
    sig += gauss(x, base_a*1.2, -1.75, 0.4)
    sig -= gauss(x, base_a, -1.5, 0.8)
    sig += 2*np.sin(x*np.pi)
    sig += x**2
    var_sig = np.var(sig, ddof=1)

    data = np.zeros((p, n), float)
    
    tries = 0
    zero_cap = np.inf
    for idx, bv in enumerate(sig):
        data[base_idx, idx] = bv
        zero_ok = pos_ok = neg_ok0 = neg_ok1 = half_ok = feat_ok = inter_ok = False
        while 1:
            tries += 1
            if tries >= 5000:
                if idx >= 491: break
                return iterate_data(params, base_a, cap1)
            print '\r{:> 3d} {:d} {:d} ({:d}, {:d}) {:d} {:d} {:d} ({:> 3.4f})'.format(idx, zero_ok, pos_ok, neg_ok0, neg_ok1, half_ok, feat_ok, inter_ok, zero_cap),
            if not half_ok:
                data[half_idx, idx] = np.random.normal(bv*0.3, 0.5)
            if not pos_ok:
                data[pos_idx, idx] = np.random.normal(bv, 0.8)
            if not zero_ok:
                data[zero_idx, idx] = np.random.normal(0, 3)
            if not neg_ok0 or not neg_ok1:
                data[neg_idx, idx] = np.random.normal(-bv, 0.8) 
            if not feat_ok:
                data[feat_idx, idx] = np.sin(idx * 6.5*np.pi/500) * base_a/2 + np.random.normal(0, 2) + gauss(idx, base_a/3.0, 1, 0.2)
            if not inter_ok:
                data[inter_idx, idx] = np.random.normal((data[feat_idx,idx]+data[half_idx,idx])/2.0, 3) 

            cov = np.cov(data)
            base_var = cov[base_idx,base_idx]
            half_var = cov[half_idx,half_idx]
            zero_var = cov[zero_idx,zero_idx]
            feat_var = cov[feat_idx,feat_idx]

            dcap = -((base_var-cap1)/float(n-1))
            zero_cap = dcap * idx + base_var
            soft_cap = zero_cap * 100

            zero_ok = -zero_cap < cov[base_idx,zero_idx] < zero_cap
            pos_ok = cov[base_idx,pos_idx] >= 0.99 * base_var
            var_zn = cov[neg_idx,neg_idx] * zero_cap
            neg_ok0 = cov[base_idx,neg_idx] <= -0.99 * base_var
            neg_ok1 = -zero_cap < cov[neg_idx,zero_idx] < zero_cap
            half_ok = 0.2 * base_var < cov[base_idx,half_idx] < 0.3 * base_var
            feat_ok = -soft_cap < cov[feat_idx,zero_idx] < soft_cap and -soft_cap < cov[feat_idx,pos_idx] < soft_cap and -soft_cap < cov[feat_idx,neg_idx] < soft_cap
            inter_ok = cov[inter_idx,half_idx] > 0.95 * half_var - (zero_cap/2.0) and 0.7 * feat_var < cov[inter_idx,feat_idx] < 0.8 * feat_var and cov[inter_idx,zero_idx] < soft_cap

            if zero_ok and pos_ok and neg_ok0 and neg_ok1 and half_ok and feat_ok and inter_ok:
                break
    return data

def gauss(x, a, b, c):
    return a * np.exp(-(x-b)**2 / c**2)

with open('./data/mock_data_pca_rnd.csv', 'w') as df:
    with open('./data/mock_data_pca_sk.csv', 'w') as df2:
        n = 500
        params = ['base', 'zero', 'pos', 'neg', 'half', 'feat', 'inter']
        base_a = 5
        p = len(params)
        fmt = '{:.2f}'.format
        cap1 = 0.0001

        data = iterate_data(params, base_a, cap1)
        
        cov = np.cov(data)
        print
        for r in cov:
            print ' '.join(map('{:> 3.5f}'.format, r))

        f, ax = plt.subplots(6,6)
        for i in xrange(6):
            for j in xrange(6):
                if i == 0:
                    ax[i,j].set_title(params[j])
                if j == 0:
                    ax[i,j].set_ylabel(params[i])

                ax[i,j].scatter(data[i], data[j], edgecolor='none', s=1)
                ax[i,j].set_xlabel('{:.5f}'.format(cov[i,j]))

        f.tight_layout()

        f2, ax = plt.subplots(3,3)
        ax[0,0].plot(data[0])
        ax[0,1].plot(data[1])
        ax[0,2].plot(data[2])
        ax[1,0].plot(data[3])
        ax[1,1].plot(data[4])
        ax[1,2].plot(data[5])
        ax[2,0].plot(data[6])
        
        d0, d1, d2 = 0, 2, 5
        fig = plt.figure()
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.scatter(data[d0], data[d1], data[d2])
        ax3d.set_xlabel(params[d0])
        ax3d.set_ylabel(params[d1])
        ax3d.set_zlabel(params[d2])

        data = data.T
        
        df.write('properties;0\n')
        df.write('parameters;{};{}\n'.format(len(params), ';'.join(params)))
        df2.write('{}\n'.format(';'.join(params)))
        for i in xrange(n):
            df.write(';'.join(map(fmt, data[i])) + '\n')
            df2.write(';'.join(map(fmt, data[i])) + '\n')

        plt.show()
