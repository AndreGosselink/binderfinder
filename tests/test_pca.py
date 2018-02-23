import pytest
from . import get_params, get_testdata
from binderfinder import PCA
import numpy as np


class TestPCA(object):
    """Unfortionally tests written after coding (and it shows a bit)
    """

    def compare_dat(self, d0, d1):
        z0 = np.argsort(d0[:,0])
        z1 = np.argsort(d1[:,0])
        return np.all(d0[z0,:] == d1[z1,:])

    def test_parsing(self):
        PCA(**get_params('pca_min'))

    def test_consistency_eigenvecs(self):
        class HookedPCA(PCA):
            def _check_consistency(selfPCA, *args, **kwargs):
                # manipulate data to make consitency check fail
                selfPCA._eigenvecs = np.random.random(selfPCA._eigenvecs.shape)
                # call PCA._check_consistency
                return super(selfPCA.__class__, selfPCA)._check_consistency(*args, **kwargs)

        with pytest.raises(ValueError):
            HookedPCA(**get_params('pca_min'))

    def test_consistency_eigenvals(self):
        class HookedPCA(PCA):
            def _check_consistency(selfPCA, *args, **kwargs):
                # manipulate data to make consitency check fail
                selfPCA._eigenvals = np.random.random(selfPCA._eigenvals.shape)
                # call PCA._check_consistency
                return super(selfPCA.__class__, selfPCA)._check_consistency(*args, **kwargs)

        with pytest.raises(ValueError):
            HookedPCA(**get_params('pca_min'))

    def test_parsing(self):
        class HookedPCA(PCA):
            def _standardize_data(selfPCA, data):
                # the testing
                assert self.compare_dat(selfPCA.data, get_testdata())
                return super(selfPCA.__class__, selfPCA)._standardize_data(data)

        HookedPCA(**get_params('pca_min'))

    def test_standard(self):
        p = PCA(**get_params('pca_min'))

        testdat = get_testdata()
        testdat -= np.mean(testdat, axis=1).reshape(testdat.shape[0], 1)
        testdat /= np.max(testdat)

        self.compare_dat(p.data, testdat)

