import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


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


tasks = ["close_drawer", "pick_up_cup", "unplug_charger", "water_plants", "wipe_desk"]
data = [[], []]
for task_name in tasks:

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle(task_name)
    ax1.set_title("Joint")
    mat_1 = np.abs(np.load("../results/correlations_context/%s_joint.npy" % task_name))
    data[0].append(average_correlation_joint(mat_1))
    print("Correlation between joints and context", average_correlation_joint(mat_1))
    ax1.imshow(rep(mat_1[:8, 8:], 20), vmin=0, vmax=1)

    ax2.set_title("Parameters")
    mat_2 = np.abs(np.load("../results/correlations_context/%s_parameter.npy" % task_name))
    data[1].append(average_correlation_parameters(mat_2))
    print("Correlation in parameter-space", average_correlation_parameters(mat_2))
    ax2.imshow(mat_2[:160, 160:], vmin=0, vmax=1)
    plt.savefig("../plots/correlations_context/%s.pdf" % task_name)
    plt.show()


X = np.arange(len(tasks))
fig, ax = plt.subplots(1, 1)
ax.bar(X + 0.00, data[0], color='b', width=0.25, label="Correlation between context-joints")
ax.bar(X + 0.25, data[1], color='g', width=0.25, label="Correlation between context-parameters")
ax.set_xticks(X + 0.125)
ax.set_xticklabels(tasks, rotation=45, ha="right")
ax.legend(loc="best")
plt.savefig("../plots/correlations_context/summary.pdf")
plt.savefig("../plots/correlations_context/summary.png")
plt.show()