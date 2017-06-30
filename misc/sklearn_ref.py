import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from matplotlib import pyplot as plt
import numpy as np
import IPython as ip

## import data

# my_csv = './data/mock_data_script_sk.csv' ## path to your dataset
my_csv = '../data/iris_dataset/iris_sk.data' ## path to your dataset
# my_csv = './data/mock_data_pca_sk.csv' ## path to your dataset

dat = pd.read_csv(my_csv, sep=';', usecols=xrange(4))
# if no row or column titles in your csv, pass 'header=None' into read_csv
# and delete 'index_col=0' -- but your biplot will be clearer with row/col names

## perform PCA

n = len(dat.columns)

pca = PCA(n_components = n)
# defaults number of PCs to number of columns in imported data (ie number of
# features), but can be set to any integer less than or equal to that value

pca.fit(dat)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
kpca.fit(dat)

## project data into PC space

# 0,1 denote PC1 and PC2; change values for other PCs
xvector = pca.components_[0] # see 'prcomp(my_data)$rotation' in R
yvector = pca.components_[1]

xs = pca.transform(dat)[:,0] # see 'prcomp(my_data)$x' in R
ys = pca.transform(dat)[:,1]

kxs = kpca.transform(dat)[:,0] # see 'prcomp(my_data)$x' in R
kys = kpca.transform(dat)[:,1]

## visualize projections
    
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them

for i in range(len(xvector[:n])):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
             list(dat.columns.values)[i], color='r')

# for i in range(len(xs)):
# # circles project documents (ie rows from csv) as points onto PC axes
#     plt.plot(xs[i], ys[i], 'bo')
#     # plt.text(xs[i]*1.2, ys[i]*1.2, list(dat.index)[i], color='b')

plt.scatter(xs, ys)

f, ax2 = plt.subplots(1)
ax2.scatter(kxs, kys)
ax2.set_title('kernel PCA')

plt.show()
