import numpy as np
from typing import List

from romi.movement_primitives import MovementPrimitive, MovementSpace, LearnTrajectory, Group, ClassicSpace, GoToTrajectory
from romi.trajectory import NamedTrajectoryBase


from .dr_interface import DimensionalityReduction

# TODO: use noise parameter
# TODO: incorporate the context


class JointReduction(DimensionalityReduction):

    def __init__(self, space: MovementSpace, dimensionality_reduction: DimensionalityReduction):
        reduced_group = Group("reduced_%s" % space.group.group_name,
                              ["rd_%d" % i for i in range(dimensionality_reduction.get_latent_dim())])
        self._observation_space = space
        self._reduced_space = ClassicSpace(reduced_group, space.n_features)
        self._dimensionality_reduction = dimensionality_reduction

    def fit(self, trajectories: List[NamedTrajectoryBase]):
        """
        Fit a
        :param trajectories:
        :return:
        """

        dataset = np.concatenate([t.values for t in trajectories], axis=0)
        self._dimensionality_reduction.fit(dataset)

    def compress(self, observed: NamedTrajectoryBase):
        X = observed.values
        t = observed.duration

        R = self.compress(X)

        trajectory = NamedTrajectoryBase(self._reduced_space.group.refs, t, R)
        return LearnTrajectory(self._reduced_space, trajectory)

    def reconstruct(self, movement: MovementPrimitive, frequency=200, duration=10., noise=False):
        trajectory = movement.get_full_trajectory(frequency, duration)
        X = self._dimensionality_reduction.reconstruct(trajectory.values)
        return NamedTrajectoryBase(self._observation_space.group.refs, trajectory.duration, X)


class PrincipalMovementPrimitive(MovementPrimitive):

    def __init__(self, movement_space: MovementSpace, parameters, dimensionality_reduction):
        """
        :param movement_space: the movement space of the movement primitive
        :type movement_space: MovementPrimitive
        :param parameters: a dictionary of the parameters for each dimension
        :type parameters: dict[np.nd_array]
        """
        block_parameters = dimensionality_reduction.reconstruct(parameters)
        params = {ref: block_parameters[movement_space.n_features * i:movement_space.n_features * (i + 1)]
                for i, ref in enumerate(movement_space.group.refs)}
        MovementPrimitive.__init__(self, movement_space, params)
        self._dimensionality_reduction = dimensionality_reduction
        self._latent_params = parameters


class ParameterReduction(DimensionalityReduction):

    def __init__(self, space: MovementSpace, dimensionality_reduction: DimensionalityReduction):
        self._observation_space = space
        self._dimensionality_reduction = dimensionality_reduction

    def fit(self, trajectories: List[NamedTrajectoryBase]):
        """
        Fit a
        :param trajectories:
        :return:
        """

        dataset = np.array([LearnTrajectory(self._observation_space, t).get_block_params() for t in trajectories])
        self._dimensionality_reduction.fit(dataset)

    def compress(self, observed: NamedTrajectoryBase):
        movement = LearnTrajectory(self._observation_space, observed)
        compress_parameters = self._dimensionality_reduction.compress(np.array([movement.get_block_params()]))
        return PrincipalMovementPrimitive(self._observation_space, compress_parameters)

    def reconstruct(self, movement: MovementPrimitive, frequency=200, duration=10., noise=False):
        return movement.get_full_trajectory(frequency, duration)