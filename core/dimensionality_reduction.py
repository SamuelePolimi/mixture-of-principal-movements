import numpy as np
from mppca.mixture_ppca import MPPCA as ExternalMPPCA
from sklearn.decomposition import PCA as ExternalPCA, FastICA

from .dr_interface import DimensionalityReduction

import torch
import torch.nn as nn
import torch.optim as optim


class ICA(DimensionalityReduction, FastICA):
    """
    TODO: understand if sk-learn is actually implementing PPCA instead of PCA
    """

    def __init__(self, *args, **kwargs):
        DimensionalityReduction.__init__(self)
        FastICA.__init__(self, *args, **kwargs)

    def get_latent_dim(self):
        return self.n_components

    def get_observed_dim(self):
        return self.n_features

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
        return self.n_components

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


class AENet(nn.Module):
    def __init__(self, input_shape, n_components, hidden_dims=[30, 20]):
        super(AENet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], n_components),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_components, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_shape)
        )

    def encode(self, X):
        return self.encoder(X)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, X):
        z = self.encoder(X)
        return self.decoder(z)

class AE:
    def __init__(self, n_components, hidden_dims=[5, 4], batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_components = n_components
        self.hidden_dims = hidden_dims

        self.net = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        self.batch_size = batch_size

        self.losses = []

    def epoch(self, X):
        indices = torch.randperm(X.size()[0])
        ep_loss = 0
        for i in range(0, X.size()[0], self.batch_size):
            self.optimizer.zero_grad()
            idx = indices[i:i+self.batch_size]
            batch = X[idx]
            reconstructions = self.net(batch)
            loss = self.criterion(reconstructions, batch)
            loss.backward()
            self.optimizer.step()
            ep_loss += loss.item()
        return ep_loss / X.shape[0]

    def fit(self, X, n_epochs=10):
        self.net = AENet(X.shape[1], self.n_components, self.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters())

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.losses = []
        for i in range(n_epochs):
            self.losses.append(self.epoch(X_tensor))
            print(i)

    def encode(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        latent = self.net.encoder(X_tensor)
        return latent.cpu().detach().numpy()

    def decode(self, z):
        z_tensor = torch.FloatTensor(z).to(self.device)
        reconstruction = self.net.decoder(z_tensor)
        return reconstruction.cpu().detach().numpy()


class Autoencoder(DimensionalityReduction, AE):
    def __init__(self, *args, **kwargs):
        DimensionalityReduction.__init__(self)
        AE.__init__(self, kwargs['n_components'])
        self.n_components = kwargs['n_components']

    def get_latent_dim(self):
        return self.n_components

    def get_observed_dim(self):
        return self.n_features_

    def fit(self, X: np.ndarray):
        self.n_features_ = X.shape[1]
        return AE.fit(self, X)

    def reconstruct(self, latent, noise=False):
        return AE.decode(self, latent)

    def compress(self, observed: np.ndarray):
        return AE.encode(self, observed)








