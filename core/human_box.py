import numpy as np

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

from .task_interface import TaskInterface
from romi.movement_primitives import ClassicSpace, MovementPrimitive, LearnTrajectory
from romi.groups import Group
from romi.trajectory import NamedTrajectory, LoadTrajectory

_name_dicts = {
    "ReachTarget": "reach_target",
    "CloseDrawer": "close_drawer",
    "WaterPlants": "water_plants",
    "PickUpCup": "pick_up_cup",
    "UnplugCharger": "unplug_charger",
    "WipeDesk": "wipe_desk",
}


class MoCap(TaskInterface):

    def __init__(self, task_name, headless=True):

        super().__init__()
        self._name = task_name
        self._group = Group("human",  ['root0', 'root1', 'root2', 'root3', 'root4', 'root5', 'lowerback0', 'lowerback1',
                                       'lowerback2', 'upperback0', 'upperback1', 'upperback2', 'thorax0', 'thorax1',
                                       'thorax2', 'lowerneck0', 'lowerneck1', 'lowerneck2', 'upperneck0', 'upperneck1',
                                       'upperneck2', 'head0', 'head1', 'head2', 'rclavicle0', 'rclavicle1', 'rhumerus0',
                                       'rhumerus1', 'rhumerus2', 'rradius0', 'rwrist0', 'rhand0', 'rhand1', 'rfingers0',
                                       'rthumb0', 'rthumb1', 'lclavicle0', 'lclavicle1', 'lhumerus0', 'lhumerus1',
                                       'lhumerus2', 'lradius0', 'lwrist0', 'lhand0', 'lhand1', 'lfingers0', 'lthumb0',
                                       'lthumb1', 'rfemur0', 'rfemur1', 'rfemur2', 'rtibia0', 'rfoot0', 'rfoot1',
                                       'rtoes0', 'lfemur0', 'lfemur1', 'lfemur2', 'ltibia0', 'lfoot0', 'lfoot1',
                                       'ltoes0'])
        self._space = ClassicSpace(self._group)

        self._state_dim = 0
        self._headless = headless

        self._obs = None

    def get_group(self):
        return self._group

    def get_context_dim(self):
        return self._state_dim

    def get_dof(self):
        return len(self._group.refs)

    def read_context(self):
        raise Exception("There is no context")

    def get_demonstrations(self, n: int):
        """
        Returns a list of trajectories, and the relative contexts!
        :param n:
        :return:
        """
        trajectory_files = ["../datasets/motion_capture/%s/trajectory_%d.npy" % (self._name, i) for i in range(n)]
        # context_files = ["../datasets/rl_bench/%s/context_%d.npy" % (self._name, i) for i in range(n)]
        try:
            return [LoadTrajectory(file) for file in trajectory_files], [np.zeros(self.get_context_dim())
                                                                         for _ in range(n)]
        except:
            raise Exception("Error in loading files!")

    def send_movement(self, movement, duration):
        raise Exception("Not Implemented!")

    def reset(self):
        raise Exception("Not Implemented")

