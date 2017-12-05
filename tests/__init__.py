import numpy as np


def get_params(kw):
    param_dict = dict(pca_min=dict(filename=r'.\data\mock_data_pca_testcases.csv',
                                   annotate=False,
                                   normalize=True,
                                   covplot=False,
                                   portions=False,
                                  ),
                      pca=dict(filename=r'.\data\mock_data_pca_rnd.csv',
                               annotate=False,
                               normalize=True,
                               covplot=False,
                               portions=False,
                              )
                     )

    return param_dict[kw]

def get_testdata():
    ret = np.array([[7.7, 5.0, 9.1, -7.7,2.2,-0.8,-0.67],
                    [8.1,-1.9, 7.4, -8.1,2.3, 2.4, 1.76],
                    [8.0,-2.2, 9.1, -7.8,2.5, 0.3, 1.77],
                    [8.3, 2.8, 9.0, -9.9,2.4,-0.0, 1.52],
                    [8.3,-4.7, 9.1, -8.7,2.0, 0.4, 0.01],
                    [9.1, 0.4, 9.6, -9.5,2.2, 1.9, 0.24],
                    [9.3,-4.4,10.2, -9.7,2.3, 5.3, 3.73],
                    [9.1, 3.9,10.3, -8.5,2.8, 2.5, 3.14],
                    [9.2, 1.0, 9.9, -9.5,2.2, 1.9, 2.06],
                    [9.3, 2.1,10.8, -8.6,2.3,-2.8,-0.02],
                    [9.2, 1.8, 9.2, -8.9,3.5, 1.2, 3.09],
                    [9.6,-3.1,10.3,-10.6,2.6, 0.6, 0.77],
                    [9.3,-1.2, 8.9, -9.0,3.9, 4.0, 2.07],
                    [9.9,-4.6,10.8,-11.7,2.5, 5.1, 3.82],
                    [9.5, 2.7, 9.2,-10.0,2.3, 0.8, 1.85],
                    [9.3,-3.5, 9.7, -9.0,2.6, 3.9, 2.88],
                    [9.8, 2.6, 9.7,-10.4,2.5, 5.6, 4.00],])
    return ret.T
