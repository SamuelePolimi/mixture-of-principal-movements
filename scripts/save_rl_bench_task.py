import os
import numpy as np
import argparse

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

from romi.movement_primitives import Group
from romi.trajectory import NamedTrajectory

from core.config import config
from core.fancy_print import f_print, PTYPE


def create_folder(path):
    try:
        os.makedirs(path)
    except OSError:
        f_print("The directory exists. Files will be overwritten.", PTYPE.warning)
        return False
    else:
        f_print("Successfully created the directory %s." % path, PTYPE.ok_green)
        return True


def collect_rl_bench_trajectories(task_class, n_movements):
        """
        Learn the Movement from demonstration.

        :param task_class: Task that we aim to learn
        :param n_features: Number of RBF in n_features
        :param n_movements: how many movements do we want to learn
        """

        frequency = 200
        group = Group("rlbench", ["d%d" % i for i in range(7)] + ["gripper"])
        # To use 'saved' demos, set the path below, and set live_demos=False
        live_demos = True
        DATASET = '' if live_demos else 'datasets'

        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)

        action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)

        env = Environment(
            action_mode, DATASET, obs_config, headless=True)
        env.launch()

        task = env.get_task(task_class)
        task_name = task.get_name()

        trajectories = []
        states = []

        lengths = []

        f_print("Start Simulation", PTYPE.ok_green)
        demos = task.get_demos(n_movements, live_demos=live_demos)
        f_print("End Simulations", PTYPE.ok_green)

        init = True
        for n_d, demo in enumerate(demos):
            trajectory = NamedTrajectory(*group.refs)
            t = 0
            for ob in demo:
                if t == 0:
                    if init:
                        print("State dim: %d" % ob.task_low_dim_state.shape[0])
                        init = False
                    states.append(ob.task_low_dim_state)
                kwargs = {"d%d" % i: ob.joint_positions[i] for i in range(ob.joint_positions.shape[0])}
                kwargs["gripper"] = ob.gripper_open
                trajectory.notify(duration=1/frequency,
                                  **kwargs)
                t += 1
            lengths.append(t/200.)
            trajectories.append(trajectory)
            f_print("DEMO %d" % n_d, PTYPE.ok_blue)

        create_folder("datasets/rl_bench/%s" % task_name)
        for i, trajectory in enumerate(trajectories):
            trajectory.save("datasets/rl_bench/%s/trajectory_%d" % (task_name, i))
        for i, state in enumerate(states):
            if i == 0:
                print("Observation dim", state.shape[0])
            np.save("datasets/rl_bench/%s/context_%d.npy" % (task_name, i), state)

        env.shutdown()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_name",
                        help="Task name.",
                        default="reach_target")
    parser.add_argument("-n", "--n-demos",
                        help="How many episodes you want to collect.",
                        type=int, default=200)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    collect_rl_bench_trajectories(config[args.task_name]["task_class"], args.n_demos)

