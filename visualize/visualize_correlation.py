import numpy as np
import matplotlib.pyplot as plt

task_name = "reach_target"
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.suptitle(task_name)
ax1.set_title("Joint")
mat_1 = np.abs(np.load("results/correlations/%s_joint.npy" % task_name))
print(np.mean(mat_1))
ax1.imshow(mat_1, vmin=0, vmax=1)

ax2.set_title("Parameters")
mat_2 = np.abs(np.load("results/correlations/%s_parameter.npy" % task_name))
print(np.mean(mat_2))
ax2.imshow(mat_2, vmin=0, vmax=1)
plt.show()