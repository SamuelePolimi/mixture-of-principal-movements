from romi.movement_primitives import Movement


class TaskInterface:

    def __init__(self):
        """
        Initialize the environment.
        """
        pass

    def get_context_dim(self):
        """
        Returns the dimension of the context variable.
        :return:
        """
        pass

    def get_dof(self):
        """
        Get the number of degrees of freedom
        :return:
        """
        pass

    def send_trajectory(self, movement: Movement, duration):
        """
        Send the movement to the robot within its duration (which can be modified internally).
        """
        pass

    def read_context(self):
        """
        Read the context (before sending the movement).
        :return:
        """
        pass

    def reset(self):
        """
        Reset the environment to a (random) initial position.
        :return:
        """
        pass

    def get_demonstrations(self, n: int):
        """
        Retrieve the matrix c
        """
        pass
