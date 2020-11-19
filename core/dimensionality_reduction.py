import numpy as np
from mppca.mixture_ppca import MPPCA as ExternalMPPCA
from sklearn.decomposition import PCA as ExternalPCA, FastICA

from .dr_interface import DimensionalityReduction


class ICA(DimensionalityReduction, FastICA):
    """
    TODO: understand if sk-learn is actually implementing PPCA instead of PCA
    """

    def __init__(self, *args, **kwargs):
        DimensionalityReduction.__init__(self)
        FastICA.__init__(self, *args, **kwargs)

    def get_latent_dim(self):
        return self.n_components_

    def get_observed_dim(self):
        return self.n_features_

    def fit(self, X: np.ndarray):
        return FastICA.fit(self, X)

    def reconstruct(self, latent, noise=False):
        return self.inverse_transform(latent)

    def compress(self, observed: np.ndarray):
        return self.transform(observed)


class PCA(DimensionalityReduction, ExternalPCA):
    """
    TODO: understand if sk-learn is actually implementing PPCA instead of PCA
    """

    def __init__(self, *args, **kwargs):
        DimensionalityReduction.__init__(self)
        ExternalPCA.__init__(self, *args, **kwargs)

    def get_latent_dim(self):
        return self.n_components_

    def get_observed_dim(self):
        return self.n_features_

    def fit(self, X: np.ndarray):
        return ExternalPCA.fit(self, X)

    def reconstruct(self, latent, noise=False):
        return self.inverse_transform(latent)

    def compress(self, observed: np.ndarray):
        return self.transform(observed)


class PPCA(DimensionalityReduction, ExternalMPPCA):

    def __init__(self, *args, **kwargs):
        DimensionalityReduction.__init__(self)
        ExternalMPPCA.__init__(self, n_components=1, *args, **kwargs)
        self.n_features_ = None

    def get_latent_dim(self):
        return self.latent_dimension

    def get_observed_dim(self):
        return self.n_features_

    def fit(self, X: np.ndarray):
        self.n_features_ = X.shape[1]
        return ExternalMPPCA.fit(self, X)

    def reconstruct(self, latent, noise=False):
        return np.array([self.sample_from_latent(0, lat, noise=noise) for lat in latent])

    def compress(self, observed: np.ndarray):
        ret = []
        for obs in observed:
            _, cluster, lat = self.reconstruction(obs, idx=np.array(range(self.get_observed_dim())))
            ret.append(lat)
        return np.array(ret)


class MPPCA(DimensionalityReduction, ExternalMPPCA):
    """
    TODO: understand if sk-learn is actually implementing PPCA instead of PCA
    """

    def __init__(self, *args, **kwargs):
        DimensionalityReduction.__init__(self)
        ExternalMPPCA.__init__(self, *args, **kwargs)
        self.n_features_ = None

    def get_latent_dim(self):
        return self.latent_dimension + 1    # the clustes is also part of the latent representation

    def get_observed_dim(self):
        return self.n_features_

    def fit(self, X: np.ndarray):
        self.n_features_ = X.shape[1]
        return ExternalMPPCA.fit(self, X)

    def reconstruct(self, latent, noise=False):
        return np.array([self.sample_from_latent(int(lat[0]), lat[1:], noise=noise) for lat in latent])

    def compress(self, observed: np.ndarray):
        ret = []
        for obs in observed:
            _, cluster, lat = self.reconstruction(obs, idx=np.array(range(self.get_observed_dim())))
            ret.append(np.concatenate([np.array([cluster]), lat]))
        return np.array(ret)









