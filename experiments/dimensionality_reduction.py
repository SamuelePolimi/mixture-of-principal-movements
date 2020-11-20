import numpy as np
import argparse
import os

from romi.movement_primitives import ClassicSpace

from core.config import config
from core.trajectory_analysis import correlation_joint_space, correlation_parameter_space
from core.movement_reduction import JointReduction, ParameterReduction
from core.dimensionality_reduction import PCA, ICA, PPCA, MPPCA
from core.fancy_print import f_print, PTYPE


def create_folder(path):
    try:
        os.makedirs(path)
    except OSError:
        f_print("The directory exists. Files will be overwritten.")
        return False
    else:
        f_print("Successfully created the directory %s." % path, PTYPE.ok_green)
        return True


def get_dr(dr_id, dr_class, latent_dim):
    if dr_id in ["ICA", "PCA"]:
        return dr_class(n_components=latent_dim)
    else:
        return dr_class(latent_dimension=latent_dim)


task_list = [
 "close_drawer", "water_plants", "pick_up_cup", "unplug_charger", "wipe_desk"
]

latent_dims = range(1, 7)


for task_name in task_list:

    reach_config = config[task_name]
    task_box = reach_config["task_box"](True)
    trajectories, contexts = task_box.get_demonstrations(200)

    train_trajectories, train_contexts = trajectories[:180], contexts[:180]
    test_trajectories, test_contexts = trajectories[180:], contexts[180:]

    space = ClassicSpace(task_box.get_group(), 20)

    for dr_id, dr_class in zip(["ICA", "PCA", "PPCA"], [ICA, PCA, PPCA]):

        result_joint = []
        result_parameter = []
        latent_dim_joint = []
        latent_dim_parameter = []

        for latent_dim in latent_dims:

            print("Task %s, Method %s, Latent dim %d" % (task_name, dr_id, latent_dim))
            dr = get_dr(dr_id, dr_class, latent_dim)

            joint_reduction = JointReduction(space, dr)

            joint_reduction.fit(train_trajectories)

            create_folder("results/dimensionality_reduction/%s" % task_name)

            partials = []
            for m_i in range(20):
                movement = joint_reduction.compress(test_trajectories[m_i])
                reconstructed_trajectory = joint_reduction.reconstruct(movement, frequency=200,
                                                                       duration=np.sum(test_trajectories[m_i].duration))
                min_points = min(reconstructed_trajectory.values.shape[0], test_trajectories[m_i].values.shape[0])
                partials.append(np.mean((reconstructed_trajectory.values[:min_points] -
                                         test_trajectories[m_i].values[:min_points]) ** 2))
            par_res = np.mean(partials)

            result_joint.append(par_res)
            latent_dim_joint.append(joint_reduction.get_latent_dim())

            print("[JOINT] mean squared error", par_res)

            dr = get_dr(dr_id, dr_class, latent_dim)

            parameter_reduction = ParameterReduction(space, dr)

            parameter_reduction.fit(trajectories)

            partials = []
            for m_i in range(20):
                movement = parameter_reduction.compress(test_trajectories[m_i])
                reconstructed_trajectory = parameter_reduction.reconstruct(movement, frequency=200,
                                                                       duration=np.sum(test_trajectories[m_i].duration))
                min_points = min(reconstructed_trajectory.values.shape[0], test_trajectories[m_i].values.shape[0])
                partials.append(np.mean((reconstructed_trajectory.values[:min_points] -
                                         test_trajectories[m_i].values[:min_points]) ** 2))

            par_res = np.mean(partials)

            print("[PARAMETER] mean squared error", par_res)
            result_parameter.append(par_res)
            latent_dim_parameter.append(parameter_reduction.get_latent_dim())

        np.savez("results/dimensionality_reduction/%s/%s.npz" % (task_name, dr_id),
                 **{"result_joint": result_joint,
                  "latent_dim_joint": latent_dim_joint,
                  "result_parameter": result_parameter,
                  "latent_dim_parameter": latent_dim_parameter})


