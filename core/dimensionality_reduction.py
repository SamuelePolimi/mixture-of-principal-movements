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

    def reconstruct(self, latent, minimum_error=True, noise=False):
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

    def reconstruct(self, latent, minimum_error=True, noise=False):
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

    def reconstruct(self, latent, minimum_error=True, noise=False):
        return np.array([self.sample_from_latent(0, lat, noise=noise) for lat in latent])

    def compress(self, observed: np.ndarray):
        ret = []
        for obs in observed:
            latent, cluster = self.get_latent(obs, use_mean_latent=True, noise=False)
            ret.append(latent)
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
        return np.array([self.sample_from_latent(self.get_proper_cluster_id(lat[0]), lat[1:], noise=noise)
                         for lat in latent])

    def get_proper_cluster_id(self, cluster):
        ret = int(np.round(cluster))
        if ret < 0:
            ret = 0
        elif ret > self.n_components - 1:
            ret = self.n_components - 1
        return ret

    def compress(self, observed: np.ndarray):
        ret = []
        for obs in observed:
            lat, cluster = self.get_latent(obs, use_mean_latent=True, noise=False)
            ret.append(np.concatenate([np.array([cluster]), lat]))
        return np.array(ret)


class AENet(nn.Module):
    def __init__(self, input_shape, n_components, hidden_dims=[30, 20]):
        super(AENet, self).__init__()
        if isinstance(hidden_dims, int):
            self.encoder = nn.Sequential(
                nn.Linear(input_shape, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, n_components),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(n_components, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, input_shape)
            )
        else:
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
    def __init__(self, n_components, hidden_dims=[5, 4], batch_size=64, learning_rate=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_components = n_components
        self.hidden_dims = hidden_dims

        self.net = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        self.batch_size = batch_size
        self.lr = learning_rate

        self.losses = []
        self.val_losses = []

    def epoch(self, X):
        self.net.train()
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

    def validate(self, X_val):
        self.net.eval()
        indices = torch.randperm(X_val.size()[0])
        ep_loss = 0
        for i in range(0, X_val.size()[0], self.batch_size):
            idx = indices[i:i+self.batch_size]
            batch = X_val[idx]
            reconstructions = self.net(batch)
            loss = self.criterion(reconstructions, batch)
            ep_loss += loss.item()
        return ep_loss / X_val.shape[0]

    def fit(self, X, X_val= None, n_epochs=10, early_stopping=True, verbose=False):
        self.net = AENet(X.shape[1], self.n_components, self.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), self.lr)

        if early_stopping:
            es = EarlyStopping()
            X_val = torch.FloatTensor(X[:int(0.1 * X.shape[0]), :]).to(self.device)
            X = X[int(0.1 * X.shape[0]):, :]

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.losses = []
        for i in range(n_epochs):
            self.losses.append(self.epoch(X_tensor))
            if X_val is not None:
                mse = self.validate(X_val)
                self.val_losses.append(mse)
                if not es.check(mse):
                    print('Early stopping criterion reached.')
                    break
            if verbose:
                print('Finished: ', i + 1)

    def encode(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        latent = self.net.encoder(X_tensor)
        return latent.cpu().detach().numpy()

    def decode(self, z):
        z_tensor = torch.FloatTensor(z).to(self.device)
        reconstruction = self.net.decoder(z_tensor)
        return reconstruction.cpu().detach().numpy()


class EarlyStopping:
    def __init__(self, patience=5, delta=0.1):
        self.patience = patience
        self.counter = 0
        self.best = 1
        self.delta = delta

    def check(self, error):
        continue_training = True
        if error < self.best - self.delta * self.best:
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            continue_training = False
        return continue_training


class Autoencoder(DimensionalityReduction, AE):
    def __init__(self, *args, **kwargs):
        DimensionalityReduction.__init__(self)
        AE.__init__(self, kwargs['n_components'], kwargs['hidden_dims'], kwargs['batch_size'], kwargs['learning_rate'])
        self.n_components = kwargs['n_components']
        self.n_epochs = kwargs['n_epochs']

    def get_latent_dim(self):
        return self.n_components

    def get_observed_dim(self):
        return self.n_features_

    def fit(self, X: np.ndarray):
        self.n_features_ = X.shape[1]
        return AE.fit(self, X, n_epochs=self.n_epochs)

    def reconstruct(self, latent, noise=False):
        return AE.decode(self, latent)

    def compress(self, observed: np.ndarray):
        return AE.encode(self, observed)








