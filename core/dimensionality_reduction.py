import numpy as np
from mppca.mixture_ppca import MPPCA as ExternalMPPCA
from sklearn.decomposition import PCA as ExternalPCA, FastICA

from core.dr_interface import DimensionalityReduction


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









