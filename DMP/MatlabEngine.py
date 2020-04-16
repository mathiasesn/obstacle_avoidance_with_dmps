import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MatlabEngine:
    
    def __init__(self):
        self.engine = matlab.engine.start_matlab()

    def exit(self):
        self.engine.exit()

    def load_robot(self):
        self.engine.workspace["robot"] = self.engine.loadrobot("frankaEmikaPanda", "DataFormat", "column")

    def simulate_robot(self):
        for i in range(self.q.shape[1]):
            self.engine.workspace["qi"] = matlab.double(self.q[:, i].tolist())
            self.engine.eval("show(robot, qi')")

    def plot_path(self):
        fig1 = plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.plot3D(self.positions[:, 0].ravel(), self.positions[:, 1].ravel(), self.positions[:, 2].ravel(), label='Demonstration')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
        
    def generate_path(self):
        self.engine.workspace["homeConfiguration"] = self.engine.eval("robot.homeConfiguration")
        self.engine.workspace["ik"] = self.engine.eval("inverseKinematics('RigidBodyTree', robot)")
        self.engine.workspace["weights"] = self.engine.ones(1, 6)

        self.engine.workspace["startPose"]   = self.engine.eval("trvec2tform([-0.5 0.5 0.4])*axang2tform([1 0 0 pi])")
        self.engine.workspace["startConfig"] = self.engine.eval("ik('panda_link8', startPose, weights, homeConfiguration)")

        self.engine.workspace["endPose"] = self.engine.eval("trvec2tform([0.5,0.2,0.4])*axang2tform([1 0 0 pi])")
        self.engine.workspace["endConfig"] = self.engine.eval("ik('panda_link8', endPose, weights, homeConfiguration)")

        q = self.engine.eval("trapveltraj([homeConfiguration,startConfig,endConfig],200,'EndTime',2)")
        self.engine.workspace["q"] = q
        self.q = np.array(q)

        # q[:, i] contains configuration for step i
        self.transforms      = []
        self.positions       = []
        for i in range(self.q.shape[1]):
            qi = matlab.double(self.q[:,i].tolist())
            self.engine.workspace["qi"] = qi

            Ti = self.engine.eval("getTransform(robot, qi', 'panda_link8')")
            self.engine.workspace["Ti"] = Ti
            self.transforms.append(Ti)

            Pi = self.engine.eval("Ti(1:3, 4)'")
            self.positions.append(Pi)
        self.transforms = np.array(self.transforms)
        self.positions  = np.array(self.positions).squeeze()


Matlab_Engine = MatlabEngine()
Matlab_Engine.load_robot()
Matlab_Engine.generate_path()
Matlab_Engine.plot_path()
Matlab_Engine.simulate_robot()
Matlab_Engine.exit()