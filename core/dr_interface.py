import numpy as np


class DimensionalityReduction:

    """
    A generic dimensionality reduction technique.

    """

    def fit(self, X: np.ndarray):
        raise NotImplemented()

    def get_observed_dim(self):
        raise NotImplemented()

    def get_latent_dim(self):
        raise NotImplemented()

    def compress(self, observed: np.ndarray):
        """

        :param observed: A matrix of observed values (a multivariate observation for each row of the matrix)
        :return: a latent space representation of the matrix
        """
        raise NotImplemented()

    def reconstruct(self, latent, noise=False):
        """
        Given a latent representation, reconstruct the original value (by sampling - for probabilistic methods)
        :param latent:
        :param noise:
        :return:
        """
        raise NotImplemented()

    def get_sample(self, noise=False):
        """
        Generate a sample from the learned distribution (applies only to probabilistic methods)
        :param noise:
        :return: (full observation, latent)
        """
        raise NotImplemented()

    def get_conditioned_sample(self, observed_values, indexes, noise=False):
        """
        Generate a sample from the learned distribution given some known values (applies only to probabilistic methods)
        :param observed_values: conditioning values
        :param indexes: indexes of the observed values
        :param noise:
        :return: (full observation, latent)
        """
        raise NotImplemented()
