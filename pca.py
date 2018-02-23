from binderfinder import PCA

p = PCA('./data/mock_data_pca_rnd.csv', annotate=True, normalize=False, covplot=True)
p.show()
