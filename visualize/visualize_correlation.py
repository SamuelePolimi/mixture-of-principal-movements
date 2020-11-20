import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task_name",
                    help="Task name.",
                    default="reach_target")
args = parser.parse_args()

task_name = args.task_name

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.suptitle(task_name)
ax1.set_title("Joint")
mat_1 = np.abs(np.load("results/correlations/%s_joint.npy" % task_name))
print("Correlation in joint-space", np.mean(mat_1))
ax1.imshow(mat_1, vmin=0, vmax=1)

ax2.set_title("Parameters")
mat_2 = np.abs(np.load("results/correlations/%s_parameter.npy" % task_name))
print("Correlation in parameter-space", np.mean(mat_2))
ax2.imshow(mat_2, vmin=0, vmax=1)
plt.savefig("plots/correlations/%s.pdf" % task_name)
plt.show()