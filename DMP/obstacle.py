import numpy as np
class Obstacle:
    def __init__(self, pos, radius = 0.015, discrete_steps = 15):
        """
        Creates spherical obstacle centered at 'pos' with radius 'radius'.
        Meshgrid created with amount of steps 'discrete_steps'.
        """
        self.pos = np.array(pos)
        self.r = radius
        self.discrete_steps = discrete_steps
        self.create_sphere()
        self.__cur_indx_move_traj = 0 # current indx of moving trajectory

    def create_sphere(self):
        phi     = np.linspace(0, np.pi, self.discrete_steps)
        theta   = np.linspace(0, 2*np.pi, self.discrete_steps)
        phi, theta = np.meshgrid(phi, theta)
        self.x = self.r * np.sin(phi) * np.cos(theta) + self.pos[0]
        self.y = self.r * np.sin(phi) * np.sin(theta) + self.pos[1]
        self.z = self.r * np.cos(phi)                 + self.pos[2]

    def create_trajectory(self, to, steps):
        """
        Create a linear trajectory from initial position
        to input parameters with resolution steps.
        """
        x_traj = np.linspace(self.pos[0], to[0], num=steps)
        y_traj = np.linspace(self.pos[1], to[1], num=steps)
        z_traj = np.linspace(self.pos[2], to[2], num=steps)
        return np.dstack((x_traj,y_traj,z_traj))

    def move_sphere(self, trajectory):
        self.pos = trajectory[self.__cur_indx_move_traj]
        self.create_sphere()

        if len(trajectory) - 1 > self.__cur_indx_move_traj:
            self.__cur_indx_move_traj += 1
            return False
        else:
            return True
        



# Matlab visualization of obstacle:
# https://se.mathworks.com/help/robotics/examples/check-for-environmental-collisions-with-manipulators.html