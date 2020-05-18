from __future__ import division, print_function
from dmp_position import PositionDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from obstacle import Obstacle

# test
import matplotlib.animation as animation





# MAIN FUNCTION 
if __name__ == '__main__':
    # # Load a demonstration file containing robot positions.
    # demo = np.loadtxt("demo.dat", delimiter=" ", skiprows=1)
    
    # tau = 0.002 * len(demo)
    # t = np.arange(0, tau, 0.002)
    # demo_p = demo[:, 0:3]

    # # Defining obstacle
    # sphere = Obstacle([0.575, 0.30, 0.45])
    # #sphere = Obstacle(demo_p[760])

    # N = 30  # TODO: Try changing the number of basis functions to see how it affects the output.
    # dmp = PositionDMP(n_bfs=N, alpha=48.0, obstacles=sphere)
    # dmp.train(demo_p, t, tau)

    # # Generate an output trajectory from the trained DMP
    # dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)

    # # 2D plot the DMP against the original demonstration
    # fig1, axs = plt.subplots(3, 1, sharex=True)
    # axs[0].plot(t, demo_p[:, 0], label='Demonstration')
    # axs[0].plot(t, dmp_p[:, 0], label='DMP')
    # axs[0].set_xlabel('t (s)')
    # axs[0].set_ylabel('X (m)')

    # axs[1].plot(t, demo_p[:, 1], label='Demonstration')
    # axs[1].plot(t, dmp_p[:, 1], label='DMP')
    # axs[1].set_xlabel('t (s)')
    # axs[1].set_ylabel('Y (m)')

    # axs[2].plot(t, demo_p[:, 2], label='Demonstration')
    # axs[2].plot(t, dmp_p[:, 2], label='DMP')
    # axs[2].set_xlabel('t (s)')
    # axs[2].set_ylabel('Z (m)')
    # axs[2].legend()

    # # 3D plot the DMP against the original demonstration       
    # fig2 = plt.figure(2)
    # ax = plt.axes(projection='3d')
    # ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    # ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='DMP')

    # ax.plot_surface(sphere.x, sphere.y, sphere.z,  rstride=1, cstride=1)


    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()

    #### TESTING ONLINE PLOTTING
    
    demo = np.loadtxt("demo.dat", delimiter=" ", skiprows=1)
    
    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    demo_p = demo[:, 0:3]

    # Recalculating DMP based on new space of sphere
   # sphere = Obstacle([0.575, 0.30, 0.45])
    sphere = Obstacle(demo_p[int(len(demo_p)/4),:])
   # obs_traj = np.squeeze(self.obstacles.create_trajectory([0.65,0.20,0.45], len(t)))

    obs_traj = demo_p[760:int(len(demo_p)/4),:] 
    obs_traj = obs_traj[::-1] # reversing path of DMP

    start_obs_mov = 450 # index of DMP when to start moving the obstacle


    N = 30  
    dmp = PositionDMP(n_bfs=N, alpha=48.0, obstacles=sphere)
    dmp.train(demo_p, t, tau)
  #  dmp.move_and_plot_dmp_obs(demo_p, t, tau, obs_traj, start_obs_mov)
    dmp.plot_euclidian_diff(demo_p, t, tau, obs_traj, start_obs_mov)
