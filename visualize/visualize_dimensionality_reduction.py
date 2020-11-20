import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


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

fig, axs = plt.subplots(5, 1)

for ax, task_name in zip(axs, task_list):

    ax.set_title(task_name)
    for dr_id, dr_class in zip(["ICA", "PCA", "PPCA"], [ICA, PCA, PPCA]):

        file = np.load("results/dimensionality_reduction/%s/%s.npz" % (task_name, dr_id))
        result_joint = file["result_joint"]
        ax.plot(file["latent_dim_joint"], result_joint, label=dr_id + " joint")
        result_parameter = file["result_parameter"]
        ax.plot(file["latent_dim_parameter"], result_parameter, label=dr_id + " parameter")

    if task_name=="close_drawer":
        ax.legend(loc="best")

plt.show()
