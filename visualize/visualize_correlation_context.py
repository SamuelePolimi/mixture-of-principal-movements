import numpy as np
import matplotlib.pyplot as plt
import argparse


def average_correlation_joint(mat):
    n = mat.shape[0]
    return np.mean(mat[:8, 8:])


def average_correlation_parameters(mat):
    n = mat.shape[0]
    return np.mean(mat[:160, 160:])


def rep(mat, n):
    rows = []
    for row in mat:
        for _ in range(n):
            rows.append(row)
    return np.array(rows)


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task_name",
                    help="Task name.",
                    default="reach_target")
args = parser.parse_args()

task_name = args.task_name

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.suptitle(task_name)
ax1.set_title("Joint")
mat_1 = np.abs(np.load("../results/correlations_context/%s_joint.npy" % task_name))
print("Correlation between joints and context", average_correlation_joint(mat_1))
ax1.imshow(rep(mat_1[:8, 8:], 20), vmin=0, vmax=1)

ax2.set_title("Parameters")
mat_2 = np.abs(np.load("../results/correlations_context/%s_parameter.npy" % task_name))
print("Correlation in parameter-space", average_correlation_parameters(mat_2))
ax2.imshow(mat_2[:160, 160:], vmin=0, vmax=1)
plt.savefig("../plots/correlations_context/%s.pdf" % task_name)
plt.show()
