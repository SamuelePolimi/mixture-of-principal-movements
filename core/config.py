from rlbench.tasks import ReachTarget, CloseDrawer, WaterPlants, PickUpCup, WipeDesk, UnplugCharger
from core.rl_bench_box import RLBenchBox, Reacher2D
from core.lab_connection import TCPTask

config = {
    "tcp": {
        "task_box": lambda headless: TCPTask(5056, 20),
        "n_features": 20,
        "n_cluster": 4,
        "latent_dim": 2,
        "state_dim": 3,
        "n_samples": 50
    },
    "tcp_pouring": {
        "task_box": lambda headless: TCPTask(5056, 20),
        "n_features": 20,
        "n_cluster": 4,
        "latent_dim": 4,
        "state_dim": 1,
        "n_samples": 100
    },
    "reacher2d_1": {
        "task_box": lambda headless: Reacher2D(20, 1, headless),
        "n_features": 20,
        "n_cluster": 1,
        "latent_dim": 2,
        "state_dim": 2,
        "n_samples": 100
    },
    "reacher2d_2": {
        "task_box": lambda headless: Reacher2D(20, 2, headless),
        "n_features": 20,
        "n_cluster": 2,
        "latent_dim": 2,
        "state_dim": 2,
        "n_samples": 100
    },
    "reacher2d_3": {
        "task_box": lambda headless: Reacher2D(20, 3, headless),
        "n_features": 20,
        "n_cluster": 3,
        "latent_dim": 2,
        "state_dim": 2,
        "n_samples": 100
    },
    "reacher2d_4": {
        "task_box": lambda headless: Reacher2D(20, 4, headless),
        "n_features": 20,
        "n_cluster": 4,
        "latent_dim": 2,
        "state_dim": 2,
        "n_samples": 100
    },
    "close_drawer": {
        "task_class": CloseDrawer,  # TODO: remove
        "task_box": lambda headless: RLBenchBox(CloseDrawer, 94, headless),
        "n_cluster": 10,
        "latent_dim": 4,
        "n_features": 20,   # TODO: remove
        "state_dim": 94,    # TODO: remove
        "n_samples": 200
    },
    "water_plants": {
        "task_class": WaterPlants,  # TODO: remove
        "task_box": lambda headless: RLBenchBox(WaterPlants, 84, headless),
        "n_cluster": 2,
        "latent_dim": 10,
        "n_features": 20,  # TODO: remove
        "state_dim": 84,  # TODO: remove
        "n_samples": 1000
    },
    "reach_target": {
        "task_class": ReachTarget,
        "task_box": lambda headless: RLBenchBox(ReachTarget, 3, headless),
        "n_cluster": 6,
        "latent_dim": 3,
        "n_features": 20,
        "state_dim": 3,
        "n_samples": 1000
    },
    "pick_up_cup": {
        "task_class": PickUpCup,
        "task_box": lambda headless: RLBenchBox(PickUpCup, 56, headless),
        "n_cluster": 6,
        "latent_dim": 3,
        "n_features": 20,
        "state_dim": 56,
        "n_samples": 1000
    },
    "wipe_desk": {
        "task_class": WipeDesk,
        "task_box": lambda headless: RLBenchBox(WipeDesk, None, headless),
        "n_cluster": 6,
        "latent_dim": 3,
        "n_features": 20,
        "state_dim": None,
        "n_samples": 1000
    },
    "unplug_charger": {
        "task_class": UnplugCharger,
        "task_box": lambda headless: RLBenchBox(UnplugCharger, None, headless),
        "n_cluster": 6,
        "latent_dim": 3,
        "n_features": 20,
        "state_dim": None,
        "n_samples": 1000
    }
}
