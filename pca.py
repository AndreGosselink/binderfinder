from binderfinder import PCA

p = PCA('./data/mock_data_pca_rnd.csv', normalize=False, debug=True)
# p = PCA('./data/iris_dataset/iris.data', debug=True, normalize=False)
# p = PCA('./data/mock_data_script.csv', debug=True, reduce_to=-1, normalize=True)
p.show()
