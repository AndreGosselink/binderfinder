from binderfinder import PCA

# p = PCA('./data/mock_data_pca_rnd.csv', annotate=True, normalize=False, reduce_to=-1)
# p = PCA('./data/iris_dataset/iris.data', annotate=True, normalize=False)
# p = PCA('./data/mock_data_script.csv')
p = PCA('./data/data.csv', normalize=True, debug=False, annotate=False)
p.show()
