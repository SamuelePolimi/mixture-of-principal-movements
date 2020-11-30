import numpy as np

from romi.movement_primitives import ClassicSpace

from core.config import config
from core.movement_reduction import JointReduction, ParameterReduction
from core.dimensionality_reduction import PCA, ICA, PPCA, MPPCA

from core.dimensionality_reduction import Autoencoder


reach_config = config["reach_target"]
task_box = reach_config["task_box"](True)
trajectories, contexts = task_box.get_demonstrations(200)

space = ClassicSpace(task_box.get_group(), 20)


# dr = ICA(n_components=3)
# dr = PCA(n_components=3)
# dr = PPCA(latent_dimension=3)
# dr = MPPCA(latent_dimension=3, n_components=3)

dr = PCA(n_components=3)

joint_reduction = JointReduction(space, dr)

joint_reduction.fit(trajectories, contexts)
reconstructed_trajectory = joint_reduction.predict(contexts[0], frequency=200, duration=np.sum(trajectories[0].duration))

print("[JOINT] mean squared error", np.mean((reconstructed_trajectory.values - trajectories[0].values)**2))


dr = MPPCA(latent_dimension=3, n_components=9)
parameter_reduction = ParameterReduction(space, dr)

parameter_reduction.fit(trajectories, contexts)
reconstructed_trajectory = parameter_reduction.predict(contexts[0], frequency=200, duration=np.sum(trajectories[0].duration))

print("[JOINT] mean squared error", np.mean((reconstructed_trajectory.values - trajectories[0].values)**2))
