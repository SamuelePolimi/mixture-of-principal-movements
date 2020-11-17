import numpy as np
import unittest

from core.dimensionality_reduction import PCA


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
        obs = lin @ latent.T
        pca = PCA(n_components=obs_dim)
        try:
            pca.fit(obs)
            L = pca.compress(obs)
            X = pca.reconstruct(L)
            self.assertTrue(np.mean(np.abs(X - obs)) <= 0.01)
        except NotImplemented:
            self.fail("myFunc() raised ExceptionType unexpectedly!")


if __name__ == "__main__":
    unittest.main()