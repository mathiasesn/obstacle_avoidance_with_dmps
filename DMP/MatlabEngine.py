import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


class MatlabEngine:
    
    def __init__(self):
        self.engine = matlab.engine.start_matlab()

    def exit(self):
        self.engine.exit()

    def load_robot(self):
        self.engine.workspace["robot"] = self.engine.loadrobot("frankaEmikaPanda", "DataFormat", "column")
        self.engine.eval("removeBody(robot,'panda_rightfinger')")
        self.engine.eval("removeBody(robot,'panda_leftfinger')")
        self.engine.eval("removeBody(robot,'panda_hand')")
        self.engine.eval("removeBody(robot,'panda_link8')")

    def simulate_robot(self):
        skip = 1
        for i in range(0,self.q.shape[1],skip): 
            self.engine.workspace["qi"] = matlab.double(self.q[:, i].tolist())
            self.engine.eval("show(robot, qi')")

    def plot_path(self):
        self.positions       = []
        for i in range(self.q.shape[1]):
            self.engine.workspace["qi"] = matlab.double(self.q[:,i].tolist())
            self.engine.workspace["Ti"] = self.engine.eval("getTransform(robot, qi', 'panda_link7')")
            Pi = self.engine.eval("Ti(1:3, 4)'")
            self.positions.append(Pi)
        self.positions  = np.array(self.positions).squeeze()


        fig1 = plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.plot3D(self.positions[:, 0].ravel(), self.positions[:, 1].ravel(), self.positions[:, 2].ravel(), label='Demonstration')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def save_trajectory(self):
        rpy = []
        traj = []
        for i in range(self.q.shape[1]):
            self.engine.workspace["qi"] = matlab.double(self.q[:,i].tolist())
            self.engine.workspace["Ti"] = self.engine.eval("getTransform(robot, qi', 'panda_link7')")

            pi  = np.array(self.engine.eval("Ti(1:3, 4)'")).squeeze()
            rpy = np.array(self.engine.eval("tform2eul(Ti)")).squeeze()

            traj.append(np.concatenate((pi, rpy)))
            
        #print(np.array(traj).shape)
        filepath = "matlabdemo.dat"
        #np.save(filepath, np.array(traj))
        #np.array(traj).tofile(filepath)
        np.savetxt(filepath, np.array(traj))
            
        
    def generate_path(self):
        self.engine.workspace["homeConfiguration"] = self.engine.eval("robot.homeConfiguration")
        self.engine.workspace["ik"] = self.engine.eval("inverseKinematics('RigidBodyTree', robot)")
        self.engine.workspace["weights"] = self.engine.ones(1, 6)

        self.engine.workspace["startPose"]   = self.engine.eval("trvec2tform([-0.5 0.5 0.4])*axang2tform([1 0 0 pi])")
        self.engine.workspace["startConfig"] = self.engine.eval("ik('panda_link7', startPose, weights, homeConfiguration)")

        self.engine.workspace["endPose"] = self.engine.eval("trvec2tform([0.5,0.2,0.4])*axang2tform([1 0 0 pi])")
        self.engine.workspace["endConfig"] = self.engine.eval("ik('panda_link7', endPose, weights, homeConfiguration)")

        q = self.engine.eval("trapveltraj([homeConfiguration,startConfig,endConfig],6000,'EndTime',2)")
        self.engine.workspace["q"] = q
        self.q = np.array(q)

    def inverseKin(self, positions, skip_frames):
        tmp_q = []
        self.engine.workspace["pi_prev"]  = self.engine.eval("robot.homeConfiguration")
        self.engine.workspace["ik"] = self.engine.eval("inverseKinematics('RigidBodyTree', robot)")
        self.engine.workspace["weights"] = self.engine.ones(1, 6)

        self.engine.workspace["configs"] = self.engine.eval("[]")
       # tqdm_1 = tqdm(range(len(positions)), ascii=True) # Taking all positions
        tqdm_1 = tqdm(range(0, len(positions), 50), ascii=True) # Taking every 100
        for i in tqdm_1:
            tqdm_1.set_description("Converting positions to configurations...")
            self.engine.workspace["pi"] = matlab.double(positions[i,:].tolist())
            
            #self.engine.workspace["configs"] = self.engine.eval("[configs trvec2tform(pi)]")
            tmp_q.append(self.engine.eval("ik('panda_link7', trvec2tform(pi), weights, pi_prev)"))
            
            self.engine.workspace["pi_prev"] = tmp_q[i//50]
        self.q = np.array(tmp_q).squeeze().transpose()

# Matlab_Engine = MatlabEngine()

# Matlab_Engine.load_robot()
# Matlab_Engine.generate_path()

# Matlab_Engine.save_trajectory()

# Matlab_Engine.plot_path()
# Matlab_Engine.simulate_robot()
# Matlab_Engine.exit()


############### TESTING INVERSE KINEMATICS FROM POSTIONS ##########################
demo = np.loadtxt("demo.dat", delimiter=" ", skiprows=1)
demo_p = demo[:, 0:3]
Matlab_Engine = MatlabEngine()

Matlab_Engine.load_robot()
Matlab_Engine.inverseKin(demo_p)
Matlab_Engine.simulate_robot()
Matlab_Engine.exit()