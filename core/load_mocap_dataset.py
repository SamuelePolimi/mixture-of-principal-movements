import numpy as np
from romi.trajectory import LoadTrajectory
import matplotlib.pyplot as plt


npz_bdata_path = '../datasets/motion_capture/mazen_c3d/airkick_running_poses.npz'  # the path to body data
bdata = np.load(npz_bdata_path)

print('Data keys available:%s'%list(bdata.keys()))
print('Vector poses has %d elements for each of %d frames.'%(bdata['poses'].shape[1], bdata['poses'].shape[0]))
print('Vector dmpls has %d elements for each of %d frames.'%(bdata['dmpls'].shape[1], bdata['dmpls'].shape[0]))
print('Vector trams has %d elements for each of %d frames.'%(bdata['trans'].shape[1], bdata['trans'].shape[0]))
print('Vector betas has %d elements constant for the whole sequence.'%bdata['betas'].shape[0])
print('The subject of the mocap sequence is %s.'%bdata['gender'])

fId = 1# frame id of the mocap sequence
root_orientation = bdata['poses'][:, :3]
body_data = bdata['poses'][:, 3:66]
hand_data = bdata['poses'][:, 66:]

