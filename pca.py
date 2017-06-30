from binderfinder import PCA

# p = PCA('./data/mock_data_pca_rnd.csv', annotate=True, normalize=False, covplot=True)
p = PCA('./data/cd3-fitc+pbmc_wiener.csv', annotate=False, normalize=True, covplot=True, last_col='class', show_class=(25, 28))
# p = PCA('./data/mock_data_script.csv')
p.show()
