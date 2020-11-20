import numpy as np
import argparse
from romi.movement_primitives import ClassicSpace


from core.config import config
from core.trajectory_analysis import correlation_joint_space, correlation_parameter_space

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task_name",
                    help="Task name.",
                    default="reach_target")
args = parser.parse_args()

env_name = args.task_name
reach_config = config[env_name]
task_box = reach_config["task_box"](True)
trajectories, contexts = task_box.get_demonstrations(100)

space = ClassicSpace(task_box.get_group(), 20)

np.save("results/correlations/%s_joint.npy" % env_name, correlation_joint_space(trajectories))
np.save("results/correlations/%s_parameter.npy" % env_name, correlation_parameter_space(trajectories, space))

