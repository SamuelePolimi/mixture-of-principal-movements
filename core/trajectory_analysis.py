import numpy as np

from romi.movement_primitives import LearnTrajectory, MovementSpace


def correlation_parameter_space(trajectories, space: MovementSpace):
    movements = [LearnTrajectory(space, t) for t in trajectories]
    dataset = np.array([movement.get_block_params() for movement in movements])
    return np.corrcoef(dataset.T)


def correlation_parameter_space_context(trajectories, contexts, space: MovementSpace):
    movements = [LearnTrajectory(space, t) for t in trajectories]
    dataset = np.array([np.concatenate([movement.get_block_params(), context])
                        for movement, context in zip(movements, contexts)])
    return np.corrcoef(dataset.T)


def correlation_joint_space(trajectories):
    dataset = np.concatenate([t.values for t in trajectories], axis=0)
    return np.corrcoef(dataset.T)


def correlation_joint_space_context(trajectories, contexts):
    dataset = np.concatenate([[np.concatenate([v, c]) for v in t.values] for t, c in zip(trajectories, contexts)], axis=0)
    return np.corrcoef(dataset.T)