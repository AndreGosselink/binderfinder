from binderfinder import PCA


params = dict(annotate=False,
             normalize=True,
               covplot=True,
              portions=True,
              # covorder=['pH', 'pA', 'pIW', 'nMC', 'pKurt', 'pSkew'],
              covorder=['PE-AR', 'PE-A', 'PE-H', 'PE-ToF', 'PE-Skew', 'PE-Kurt'],
              class_idx=6,
             )

# p = PCA('./data/mock_data_pca_rnd.csv', annotate=True, normalize=False, covplot=True)
p = PCA(r'C:\Users\andreg\Documents\masterarbeit\data\feature_extraction\cd3-fitc+pbmc_event_data.csv', **params)
# p = PCA(r'C:\Users\andreg\Documents\masterarbeit\data\feature_extraction\cd3-fitc+pbmc_wiener.csv', **params)
# p = PCA('./data/mock_data_script.csv')
p.show()
