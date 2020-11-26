import numpy as np
import matplotlib.pyplot as plt
import argparse


def average_correlation(mat):
    n = mat.shape[0]
    return (np.sum(mat) - np.trace(mat))/(n*(n-1))


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task_name",
                    help="Task name.",
                    default="reach_target")
args = parser.parse_args()

task_name = args.task_name

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.suptitle(task_name)
ax1.set_title("Joint")
mat_1 = np.abs(np.load("../results/correlations/%s_joint.npy" % task_name))
print("Correlation in joint-space", average_correlation(mat_1))
ax1.imshow(mat_1, vmin=0, vmax=1)

ax2.set_title("Parameters")
mat_2 = np.abs(np.load("../results/correlations/%s_parameter.npy" % task_name))
print("Correlation in parameter-space", average_correlation(mat_2))
ax2.imshow(mat_2, vmin=0, vmax=1)
plt.savefig("../plots/correlations/%s.pdf" % task_name)
plt.show()