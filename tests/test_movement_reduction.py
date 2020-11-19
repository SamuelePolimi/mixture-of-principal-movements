import numpy as np
import unittest

from core.dimensionality_reduction import PCA, PPCA, MPPCA


class TestPCA(unittest.TestCase):

    def test_fit(self):
        """
        Check that the multiclass inheritance is properly working.
        :return:
        """
        obs_dim = 5
        latent_dim = 3
        latent = np.random.multivariate_normal(np.zeros(latent_dim), np.eye(latent_dim), size=1000)
        lin = np.random.uniform(size=(obs_dim, latent_dim))
        obs = latent @ lin.T
        pca = PCA(n_components=obs_dim)
        try:
            pca.fit(obs)
            L = pca.compress(obs)
            X = pca.reconstruct(L)
            self.assertTrue(np.mean(np.abs(X - obs)) <= 0.01)
        except NotImplemented:
            self.fail("myFunc() raised ExceptionType unexpectedly!")

        ppca = PPCA(latent_dimension=3)
        try:
            ppca.fit(obs)
            L = ppca.compress(obs)
            X = ppca.reconstruct(L)
            self.assertTrue(np.mean(np.abs(X - obs)) <= 0.01)
        except NotImplemented:
            self.fail("myFunc() raised ExceptionType unexpectedly!")

        mppca = MPPCA(n_components=2, latent_dimension=3)
        try:
            mppca.fit(obs)
            L = mppca.compress(obs)
            X = mppca.reconstruct(L)
            self.assertTrue(np.mean(np.abs(X - obs)) <= 0.01)
        except NotImplemented:
            self.fail("myFunc() raised ExceptionType unexpectedly!")


if __name__ == "__main__":
    unittest.main()