from binderfinder import PCA

p = PCA('./data/mock_data_pca_rnd.csv', annotate=True, normalize=False)
# p = PCA('./data/iris_dataset/iris.data', annotate=True, normalize=False)
# p = PCA('./data/mock_data_script.csv')
p.show()
