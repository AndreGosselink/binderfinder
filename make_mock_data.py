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

with open('./data/mock_data_pca_rnd.csv', 'w') as df:
    n = 500
    p = 5
    smalls = 50
    bigs = 200
    fmt = '{:.2f}'.format
    all_high = 50
    data = np.zeros((n, p))
    # gen 0-2
    data[:,0] = np.random.normal(5, 1, n)
    data[:,1] = np.random.normal(5, 1, n)
    data[:,2] = np.random.normal(5, 1, n)
    # small
    data[:smalls,3] = np.random.normal(6, 1, smalls)
    data[smalls:,3] = np.random.normal(4, 2, n-smalls)
    # big
    data[:bigs,4] = np.random.normal(10, 3, bigs)
    data[bigs:,4] = np.random.normal(25, 2.5, n-bigs)
    # all
    data[-all_high:,:] = np.random.normal(16, 0.3, (all_high, p))

    df.write('properties;0\n')
    df.write('parameters;{};gen0;gen1;gen2;small;big;all\n'.format(p))
    for i in xrange(n):
        df.write(';'.join(map(fmt, data[i])) + '\n')


f, ax = plt.subplots(2)
ax[0].scatter(data[:,0], data[:,1], edgecolor=(0, 0, 0), s=3)
ax[1].scatter(data[:,3], data[:,4], edgecolor=(0, 0, 0), s=3)
plt.show()
