from __future__ import division, print_function

import numpy as np

from canonical_system import CanonicalSystem
from obstacle import Obstacle
from repulsive import Ct, Ct_coupling

# test
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


class PositionDMP():
    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None, cs=None, obstacles=None):
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        self.cs = cs if cs is not None else CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else self.alpha/2)

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c)**2

        # Scaling factor
        self.Dp = np.identity(3)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((3, self.n_bfs))

        # Initial- and goal positions
        self.p0 = np.zeros(3)
        self.gp = np.zeros(3)

        self.obstacles = obstacles

        self.reset()

    def step(self, x, dt, tau):
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration
        # TODO: Implement the transformation system differential equation for the acceleration, given that you know the
        # values of the following variables:
        # self.alpha, self.beta, self.gp, self.p, self.dp, tau, x

        ###### OLD ######
        #sphere  = Obstacle([0.575, 0.30, 0.45])
        #sphere = Obstacle([0., 0.25, 0.80])
        ###### NEW ######

        
        self.ddp = (self.alpha*( self.beta * (self.gp - self.p) - tau*self.dp ) + fp(x) + Ct_coupling(self.p, self.dp, self.obstacles) )/(tau*tau)

        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        return self.p, self.dp, self.ddp

    # def move_and_plot_dmp_obs(self, demo_p, ts, tau, obstacle):
    #     self.reset()
    #     if np.isscalar(tau):
    #         tau = np.full_like(ts, tau)

    #     x = self.cs.rollout(ts, tau)  # Integrate canonical system
    #     dt = np.gradient(ts) # Differential time vector

    #     n_steps = len(ts)

    #     def gen_one_step(i, obs, trajectory):
    #         def fp(xj):
    #             psi = np.exp(-self.h * (xj - self.c)**2)
    #             return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

    #         self.ddp = (self.alpha*( self.beta * (self.gp - self.p) - tau[i]*self.dp ) + fp(x[i]) + Ct_coupling(self.p, self.dp, obs) )/(tau[i]*tau[i])

    #         # Integrate acceleration to obtain velocity
    #         self.dp += self.ddp * dt[i]

    #         # Integrate velocity to obtain position
    #         self.p += self.dp * dt[i]

    #         # Moving obstacle
    #         obs.move_sphere(trajectory)
            
    #         ax.clear()
    #         # ax.set_xlim(0, 1.2)
    #         # ax.set_ylim(0, 1.2)
    #         # ax.set_zlim(0, 1.2)
    #         ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    #         print(self.p[0], self.p[1], self.p[2])
    #         ax.plot3D((self.p[0],self.p[0]), (self.p[1],self.p[1]), (self.p[2],self.p[2]), label='DMP')
    #         plot3d = ax.plot_surface(obs.x, obs.y, obs.z, rstride=1, cstride=1)
    #         return plot3d,

    #     # Generate trajectory for obstacle
    #     traj = np.squeeze(obstacle.create_trajectory([0.65,0.20,0.45], n_steps))
    #     # Making animation
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')

    #     ani = animation.FuncAnimation(fig, gen_one_step, fargs=(obstacle,traj),frames=n_steps-1, interval=30, blit=False, repeat=False)
    #     plt.show()

    def move_and_plot_dmp_obs(self, demo_p, ts, tau):
        self.reset()
        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)

        # Generating the points for both obstacle and DMP
        dmp_p = np.empty((n_steps, 3))
        obs_p = []
        obs_traj = np.squeeze(self.obstacles.create_trajectory([0.65,0.20,0.45], n_steps))
        
        for i in range(n_steps):
            dmp_p[i], _, _ = self.step(x[i], dt[i], tau[i])
            obs_p.append((self.obstacles.x, self.obstacles.y, self.obstacles.z))
            # Moving obstacle
            self.obstacles.move_sphere(obs_traj)
            
        def one_step_and_animate(i, plot_dmp, dmp_points, plot_obs, obs_points):
            # Plotting DMP
            plot_dmp.set_data(dmp_points[0:i+1,0], dmp_points[0:i+1,1])
            plot_dmp.set_3d_properties(dmp_points[0:i+1,2])

            # Plotting obstacle
            plot_obs[0].remove()
            plot_obs[0] = ax.plot_surface(obs_points[i][0], obs_points[i][1], obs_points[i][2], cmap="magma")
      
        # Making animation
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        # Static plots and settings
        # ax.set_xlim(0, 1.2)
        # ax.set_ylim(0, 1.2)
        # ax.set_zlim(0, 1.2)
        ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')

        plot_dmp = ax.plot3D([], [], [])[0]
        plot_obs = [ax.plot_surface(obs_p[0][0], obs_p[0][1], obs_p[0][2], rstride=1, cstride=1)]

        ani = animation.FuncAnimation(fig, one_step_and_animate, fargs=[plot_dmp, dmp_p, plot_obs, obs_p], frames=n_steps-1, interval=25, blit=False, repeat=False)
        
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=500, bitrate=2800)
       # ani.save('im.mp4', writer=writer)
       #ani.save('sine_wave.gif', writer='imagemagick', fps=1000, dpi=100, bitrate=1800 )
        plt.show()

    def rollout(self, ts, tau):
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)
        p = np.empty((n_steps, 3))
        dp = np.empty((n_steps, 3))
        ddp = np.empty((n_steps, 3))

        for i in range(n_steps):
            p[i], dp[i], ddp[i] = self.step(x[i], dt[i], tau[i])

        return p, dp, ddp

    def reset(self):
        self.p = self.p0.copy()
        self.dp = np.zeros(3)
        self.ddp = np.zeros(3)

    def train(self, positions, ts, tau):
        p = positions

        # Sanity-check input
        if len(p) != len(ts):
            raise RuntimeError("len(p) != len(ts)")

        # Initial- and goal positions
        self.p0 = p[0]
        self.gp = p[-1]

        # Differential time vector
        dt = np.gradient(ts)[:,np.newaxis]

        # Scaling factor
        self.Dp = np.diag(self.gp - self.p0)
        Dp_inv = np.linalg.inv(self.Dp)

        # Desired velocities and accelerations
        d_p = np.gradient(p, axis=0) / dt
        dd_p = np.gradient(d_p, axis=0) / dt

        # Integrate canonical system
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return xj * psi / psi.sum()

        def forcing(j):
            return Dp_inv.dot(tau**2 * dd_p[j]
                - self.alpha * (self.beta * (self.gp - p[j]) - tau * d_p[j]))

        A = np.stack(features(xj) for xj in x)
        f = np.stack(forcing(j) for j in range(len(ts)))

        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_p = p
        self.train_d_p = d_p
        self.train_dd_p = dd_p
