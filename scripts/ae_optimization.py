import numpy as np

from romi.movement_primitives import ClassicSpace, LearnTrajectory

from core.config import config
from core.movement_reduction import JointReduction, ParameterReduction
import matplotlib.pyplot as plt

from core.dimensionality_reduction import Autoencoder, AE

reach_config = config["reach_target"]
task_box = reach_config["task_box"](True)
trajectories, contexts = task_box.get_demonstrations(200)
test_traj = trajectories[0]
trajectories = trajectories[1:]

traj_train = trajectories[:int(len(trajectories) * .8)]
traj_val = trajectories[int(len(trajectories) * .8):]

space = ClassicSpace(task_box.get_group(), 20)

joint_train_dataset = np.concatenate([t.values for t in traj_train], axis=0)
param_train_dataset = np.array([LearnTrajectory(space, t).get_block_params() for t in traj_train])
joint_val_dataset = np.concatenate([t.values for t in traj_val], axis=0)
param_val_dataset = np.array([LearnTrajectory(space, t).get_block_params() for t in traj_val])
joint_test = test_traj.values
param_test = LearnTrajectory(space, test_traj).get_block_params().reshape(1, -1)

def train_ae(X_train, X_val, X_test):
    ae = AE(3, [10, 5], 16)
    ae.fit(X_train, X_val, 10, True)
    latent_representation = ae.encode(X_test)
    reconstructed = ae.decode(latent_representation)
    error = np.mean((reconstructed - trajectories[0].values) ** 2)
    print(error)
    #plt.plot(ae.losses, label='train loss')
    #plt.plot(ae.val_losses, label='val loss')
    #plt.legend()
    #plt.show()

def joint_ae(traj_train, traj_test, n_comps, hidden_dims, bs, lr, n_epochs):
    dr = Autoencoder(n_components=n_comps, hidden_dims=hidden_dims, batch_size=bs, n_epochs=n_epochs, learning_rate=lr)
    joint_reduction = JointReduction(space, dr)

    joint_reduction.fit(traj_train)
    movement = joint_reduction.compress(traj_test)
    reconstructed_trajectory = joint_reduction.reconstruct(movement, frequency=200,
                                                           duration=np.sum(traj_test.duration))
    mse = np.mean((reconstructed_trajectory.values - traj_test.values)**2)
    return mse

def param_ae(traj_train, traj_test, n_comps, hidden_dims, bs, lr, n_epochs):
    dr = Autoencoder(n_components=n_comps, hidden_dims=hidden_dims, batch_size=bs, n_epochs=n_epochs, learning_rate=lr)
    parameter_reduction = ParameterReduction(space, dr)

    parameter_reduction.fit(traj_train)
    movement = parameter_reduction.compress(traj_test)
    reconstructed_trajectory = parameter_reduction.reconstruct(movement, frequency=200,
                                                               duration=np.sum(traj_test.duration))

    return np.mean((reconstructed_trajectory.values - traj_test.values) ** 2)

def optimize_ae(n_comps, n_configs, method):
    batch_size_range = np.linspace(1, 128, dtype=int)
    hidden_dims_1 = np.arange(n_comps, 4 * n_comps)
    hidden_dims_2 = np.arange(n_comps, 3 * n_comps)
    learning_rates = np.linspace(1e-5, 1e-2)

    scores = {}

    for c in range(n_configs):
        bs = np.random.choice(batch_size_range)
        lr = np.random.choice(learning_rates)
        hidden_dims = [np.random.choice(hidden_dims_1), np.random.choice(hidden_dims_2)]

        if method == 'joint':
            mse = joint_ae(traj_train, test_traj, n_comps, hidden_dims, bs, lr, 20)
        if method == 'param':
            mse = param_ae(traj_train, test_traj, n_comps, hidden_dims, bs, lr, 20)
        scores[mse] = {'bs': bs, 'lr': lr, 'hidden_dims': hidden_dims}
        print(c + 1)
    f_name = method + '_results_' + str(n_configs) + '_trials'
    np.save(f_name, scores)

def print_scores(f_name):
    scores = np.load(f_name, allow_pickle=True)
    sorted_keys = sorted(scores.item().keys())
    for i in range(3):
        score = sorted_keys[i]
        params = scores.item()[score]
        print(params, ' | score: ', score)
#optimize_ae(3, 50, 'joint')
print('Joint Space')
print(100 * '=')
print_scores('joint_results_50_trials.npy')
print(100 * '-')
print('Parameter Space')
print(100 * '=')
print_scores('param_results_100_trials.npy')
print(100 * '-')
#train_ae(param_train_dataset, param_val_dataset, param_test)