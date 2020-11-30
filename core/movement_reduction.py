import numpy as np
from typing import List
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from mppca.mixture_ppca import sum_logs
from romi.movement_primitives import MovementPrimitive, MovementSpace, LearnTrajectory, Group, ClassicSpace, GoToTrajectory
from romi.trajectory import NamedTrajectoryBase


from .dr_interface import DimensionalityReduction

# TODO: use noise parameter
# TODO: incorporate the context


class GaussianMixtureRegression(GaussianMixture):

    def __init__(self, *args, **kwargs):
        GaussianMixture.__init__(self, *args, **kwargs)

    def get_responsabilities(self, X: np.ndarray, idxs: np.ndarray):
        """

        :param X: Data
        :param n_components: number of the component of the mixture model
        :param means: means of the clusters
        :param covariances: covariances of the clusters
        :param log_pi: log weights of the mixture model
        :return: log_responsabilities, log_likelihood (both sample-wise)
        """

        obs_dim = self.means_[0].shape[0]
        R_log = np.zeros(self.n_components)
        P_log = np.zeros(self.n_components)

        for i in range(self.n_components):
            P_log[i] = multivariate_normal.logpdf(X, self.means_[i][idxs], self.covariances_[i][idxs][:, idxs],
                                                  allow_singular=True)

        log_scaling = sum_logs(np.array([P_log[j] + np.log(self.weights_)[j]
                                         for j in range(self.n_components)]), axis=0)

        for i in range(self.n_components):
            R_log[i] = P_log[i] + np.log(self.weights_)[i] - log_scaling  # eq 21

        return R_log, P_log

    def conditional(self, mean, covariance, X, idx, idx_query, reg_cov_bb=0.):
        indx_a = idx_query
        indx_b = idx
        mu_a = mean[indx_a]
        mu_b = mean[indx_b]
        cov_aa = covariance[indx_a][:, indx_a]
        cov_ab = covariance[indx_a][:, indx_b]
        cov_bb = covariance[indx_b][:, indx_b] + reg_cov_bb * np.eye(len(idx))

        inv_cov_bb = np.linalg.inv(cov_bb)
        mu_a_b = mu_a + cov_ab @ inv_cov_bb @ (X - mu_b)

        return mu_a_b, cov_aa - cov_ab @ inv_cov_bb @ cov_ab.T

    def regression(self, x, indxs: np.ndarray):
        obs_dim = self.means_[0].shape[0]
        r_log, _ = self.get_responsabilities(x, indxs)
        ret = 0.
        idx_query = [i for i in range(obs_dim) if i not in indxs]
        for i in range(self.n_components):
            mean, _ = self.conditional(self.means_[i], self.covariances_[i], x, indxs, idx_query)
            ret += np.exp(r_log[i]) * mean

        return ret


class JointReduction(DimensionalityReduction):

    def __init__(self, space: MovementSpace, dimensionality_reduction: DimensionalityReduction, n_cluster=1):
        reduced_group = Group("reduced_%s" % space.group.group_name,
                              ["rd_%d" % i for i in range(dimensionality_reduction.get_latent_dim())])
        self._observation_space = space
        self._reduced_space = ClassicSpace(reduced_group, space.n_features)
        self._dimensionality_reduction = dimensionality_reduction
        self._n_cluster = n_cluster

    def fit(self, trajectories: List[NamedTrajectoryBase], contexts=None):
        """
        Fit a
        :param trajectories:
        :return:
        """

        dataset = np.concatenate([t.values for t in trajectories], axis=0)
        self._dimensionality_reduction.fit(dataset)
        if contexts is not None:
            self._data_parameters = [self.compress(t).get_block_params() for t in trajectories]
            total_data = np.concatenate([self._data_parameters, contexts], axis=1)
            self._gmr = GaussianMixtureRegression(n_components=self._n_cluster)
            self._gmr.fit(total_data)

    def compress(self, observed: NamedTrajectoryBase):
        X = observed.values
        t = observed.duration

        R = self._dimensionality_reduction.compress(X)

        trajectory = NamedTrajectoryBase(self._reduced_space.group.refs, t, R)
        return LearnTrajectory(self._reduced_space, trajectory)

    def reconstruct(self, movement: MovementPrimitive, frequency=200, duration=10., noise=False):
        trajectory = movement.get_full_trajectory(frequency, duration)
        X = self._dimensionality_reduction.reconstruct(trajectory.values)
        return NamedTrajectoryBase(self._observation_space.group.refs, trajectory.duration, X)

    def predict(self, context, frequency=200, duration=10., noise=False):
        latent_joint = self._dimensionality_reduction.get_latent_dim()
        parameters = self._gmr.regression(context, indxs=[i + self.get_latent_dim()
                                                                   for i in range(context.shape[0])]).reshape(latent_joint, -1)
        movement = MovementPrimitive(self._reduced_space, {'rd_%d' % i: w for i, w in enumerate(parameters)})
        trajectory = movement.get_full_trajectory(frequency, duration)
        X = self._dimensionality_reduction.reconstruct(trajectory.values)
        return NamedTrajectoryBase(self._observation_space.group.refs, trajectory.duration, X)

    def get_latent_dim(self):
        return self._dimensionality_reduction.get_latent_dim() * self._reduced_space.n_features

    def get_observed_dim(self):
        return self._observation_space.n_params


class PrincipalMovementPrimitive(MovementPrimitive):

    def __init__(self, movement_space: MovementSpace, parameters, dimensionality_reduction):
        """
        :param movement_space: the movement space of the movement primitive
        :type movement_space: MovementPrimitive
        :param parameters: a dictionary of the parameters for each dimension
        :type parameters: dict[np.nd_array]
        """
        block_parameters = dimensionality_reduction.reconstruct(parameters).ravel()
        params = {ref: block_parameters[movement_space.n_features * i:movement_space.n_features * (i + 1)]
                for i, ref in enumerate(movement_space.group.refs)}
        MovementPrimitive.__init__(self, movement_space, params)
        self._dimensionality_reduction = dimensionality_reduction
        self._latent_params = parameters


class ParameterReduction(DimensionalityReduction):

    def __init__(self, space: MovementSpace, dimensionality_reduction: DimensionalityReduction):
        self._observation_space = space
        self._dimensionality_reduction = dimensionality_reduction

    def fit(self, trajectories: List[NamedTrajectoryBase], contexts=None):
        """
        Fit a
        :param trajectories:
        :return:
        """

        dataset = np.array([LearnTrajectory(self._observation_space, t).get_block_params() for t in trajectories])
        if contexts is None:
            self._dimensionality_reduction.fit(dataset)
        else:
            self._dimensionality_reduction.fit(np.concatenate([dataset, contexts], axis=1))

    def get_latent_dim(self):
        return self._dimensionality_reduction.get_latent_dim()

    def get_observed_dim(self):
        return self._observation_space.n_params

    def compress(self, observed: NamedTrajectoryBase):
        movement = LearnTrajectory(self._observation_space, observed)
        compress_parameters = self._dimensionality_reduction.compress(np.array([movement.get_block_params()]))
        return PrincipalMovementPrimitive(self._observation_space, compress_parameters, self._dimensionality_reduction)

    def predict(self, context, frequency=200, duration=10.):
        n = len(self._observation_space.group.refs)
        parameters = self._dimensionality_reduction.get_conditioned_sample(context, indexes=[i + self._observation_space.n_params
                                                                   for i in range(context.shape[0])]).reshape(n, -1)
        mp = MovementPrimitive(self._observation_space, {k: w for k, w in zip(self._observation_space.group.refs, parameters)})
        return mp.get_full_trajectory(frequency, duration)

    def reconstruct(self, movement: MovementPrimitive, frequency=200, duration=10., noise=False):
        return movement.get_full_trajectory(frequency, duration)