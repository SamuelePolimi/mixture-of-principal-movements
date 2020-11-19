import numpy as np

from romi.movement_primitives import ClassicSpace

from core.config import config
from core.movement_reduction import JointReduction, ParameterReduction
from core.dimensionality_reduction import PCA

reach_config = config["reach_target"]
task_box = reach_config["task_box"](True)
trajectories, contexts = task_box.get_demonstrations(100)

space = ClassicSpace(task_box.get_group(), 20)
pca = PCA(n_components=3)
joint_reduction = JointReduction(space, pca)

joint_reduction.fit(trajectories)
movement = joint_reduction.compress(trajectories[0])
reconstructed_trajectory = joint_reduction.reconstruct(movement, frequency=200, duration=np.sum(trajectories[0].duration))

print("[JOINT] mean squared error", np.mean((reconstructed_trajectory.values - trajectories[0].values)**2))

parameter_reduction = ParameterReduction(space, pca)

parameter_reduction.fit(trajectories)
movement = parameter_reduction.compress(trajectories[0])
reconstructed_trajectory = parameter_reduction.reconstruct(movement, frequency=200, duration=np.sum(trajectories[0].duration))

print("[PARAMETER] mean squared error", np.mean((reconstructed_trajectory.values - trajectories[0].values)**2))