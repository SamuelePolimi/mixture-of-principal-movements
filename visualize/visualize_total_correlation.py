import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def average_correlation(mat):
    n = mat.shape[0]
    return (np.sum(mat) - np.trace(mat))/(n*(n-1))


tasks =  ["close_drawer", "pick_up_cup", "unplug_charger", "water_plants", "wipe_desk"]
data = [[], []]
for task_name in tasks:

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle(task_name)
    ax1.set_title("Joint")
    mat_1 = np.abs(np.load("../results/correlations/%s_joint.npy" % task_name))
    data[0].append(average_correlation(mat_1))
    print("Correlation in joint-space", average_correlation(mat_1))
    ax1.imshow(mat_1, vmin=0, vmax=1)

    ax2.set_title("Parameters")
    mat_2 = np.abs(np.load("../results/correlations/%s_parameter.npy" % task_name))
    data[1].append(average_correlation(mat_2))
    print("Correlation in parameter-space", average_correlation(mat_2))
    ax2.imshow(mat_2, vmin=0, vmax=1)
    plt.savefig("../plots/correlations/%s.pdf" % task_name)
    plt.show()


X = np.arange(len(tasks))
fig, ax = plt.subplots(1, 1)
ax.bar(X + 0.00, data[0], color='b', width=0.25, label="Correlation between joints")
ax.bar(X + 0.25, data[1], color='g', width=0.25, label="Correlation between parameters")
ax.set_xticks(X + 0.125)
ax.set_xticklabels(tasks, rotation=45, ha="right")
ax.legend(loc="best")
plt.savefig("../plots/correlations/summary.pdf")
plt.savefig("../plots/correlations/summary.png")
plt.show()