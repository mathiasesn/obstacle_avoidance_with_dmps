import numpy as np
class Obstacle:
    def __init__(self, pos, radius = 0.025, discrete_steps = 30):
        """
        Creates spherical obstacle centered at 'pos' with radius 'radius'.
        Meshgrid created with amount of steps 'discrete_steps'.
        """
        self.pos = np.array(pos)
        self.r = radius
        self.discrete_steps = discrete_steps
        self.create_sphere()

    def create_sphere(self):
        phi     = np.linspace(0, np.pi, self.discrete_steps)
        theta   = np.linspace(0, 2*np.pi, self.discrete_steps)
        phi, theta = np.meshgrid(phi, theta)
        self.x = self.r * np.sin(phi) * np.cos(theta) + self.pos[0]
        self.y = self.r * np.sin(phi) * np.sin(theta) + self.pos[1]
        self.z = self.r * np.cos(phi)                 + self.pos[2]