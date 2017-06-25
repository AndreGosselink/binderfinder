from binderfinder import PCA

p = PCA('./data/mock_data_pca_rnd.csv', normalize=True)
# p = PCA('./data/iris_dataset/iris.data', debug=False)
# p = PCA('./data/mock_data_script.csv', debug=True, reduce_to=-1)
p.show()
