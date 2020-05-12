<<<<<<< HEAD
from __future__ import division, print_function

import numpy as np

from canonical_system import CanonicalSystem
from obstacle import Obstacle
from repulsive import Ct, Ct_coupling

# test
from tqdm import tqdm
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

    def rollout_moving_obstacle(self, ts, tau, obs_traj, start_obs_mov):
        self.reset()
        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)

        # Generating the points for both obstacle and DMP
        dmp_p = np.empty((n_steps, 3))
        obs_p = []
        obs_p.append((self.obstacles.x, self.obstacles.y, self.obstacles.z))

        for i in range(n_steps):
            dmp_p[i], _, _ = self.step(x[i], dt[i], tau[i])
            
            # Moving obstacle
            # First moving obstacle at index DMP
            if i > start_obs_mov:
                print(self.obstacles.pos)
                stop_mov = self.obstacles.move_sphere(obs_traj)
                if not stop_mov:
                    obs_p.append((self.obstacles.x, self.obstacles.y, self.obstacles.z))
            
        return dmp_p, obs_p

    def move_and_plot_dmp_obs(self, demo_p, ts, tau, obs_traj, start_obs_mov):
        n_steps = len(ts)
        dmp_p, obs_p = self.rollout_moving_obstacle(ts, tau, obs_traj, start_obs_mov)

        def animate(i, plot_dmp, dmp_points, plot_obs, obs_points, update_bar):
            # Updating progress bar
            update_bar(1)
            # Plotting DMP
            plot_dmp.set_data(dmp_points[0:i+1,0], dmp_points[0:i+1,1])
            plot_dmp.set_3d_properties(dmp_points[0:i+1,2])

            # Plotting obstacle
            if (len(obs_points) - 2 > self.obs_plt_indx) and (i > start_obs_mov): # Showing with magma color while moving and without when not
                plot_obs[0].remove()
                plot_obs[0] = ax.plot_surface(obs_points[self.obs_plt_indx][0], obs_points[self.obs_plt_indx][1], obs_points[self.obs_plt_indx][2], cmap="magma")
                self.obs_plt_indx += 1
            elif (len(obs_points) - 1 > self.obs_plt_indx) and (i > start_obs_mov):
                plot_obs[0].remove()
                plot_obs[0] = ax.plot_surface(obs_points[self.obs_plt_indx][0], obs_points[self.obs_plt_indx][1], obs_points[self.obs_plt_indx][2], rstride=1, cstride=1)
                self.obs_plt_indx += 1

       
        with tqdm(total=n_steps) as tbar:
            # Making animation
            fig = plt.figure()
            ax = p3.Axes3D(fig)

            # Static plots and settings
            # ax.set_xlim(0, 1.2)
            # ax.set_ylim(0, 1.2)
            # ax.set_zlim(0, 1.2)
            ax.view_init(elev=15, azim=-120)
            ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')

            plot_dmp = ax.plot3D([], [], [])[0]
            plot_obs = [ax.plot_surface(obs_p[0][0], obs_p[0][1], obs_p[0][2], rstride=1, cstride=1)]
            self.obs_plt_indx = 0

            ani = animation.FuncAnimation(fig, animate, fargs=[plot_dmp, dmp_p, plot_obs, obs_p, tbar.update], frames=n_steps-1, interval=25, blit=False, repeat=False)
            
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=2000)
            #ani.save('im.mp4', writer=writer)
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
=======
from __future__ import division, print_function

import numpy as np

from canonical_system import CanonicalSystem
from obstacle import Obstacle
from repulsive import Ct, Ct_coupling
from attractive import Att

class PositionDMP():
    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None, cs=None):
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

        self.reset()

    def step(self, x, dt, tau, x_target):
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration
        # TODO: Implement the transformation system differential equation for the acceleration, given that you know the
        # values of the following variables:
        # self.alpha, self.beta, self.gp, self.p, self.dp, tau, x
        #sphere  = Obstacle([0.575, 0.30, 0.45])
        #sphere = Obstacle([0., 0.25, 0.80])
        sphere  = Obstacle(self.demo_p[760])

       # x_target = self.p + (self.dp + self.alpha*( self.beta * (self.gp - self.p) - tau*self.dp ) / (tau * tau) * dt)*dt # target used for Att

        self.ddp = (self.alpha*( self.beta * (self.gp - self.p) - tau*self.dp ) + fp(x) + Ct_coupling(self.p, self.dp, sphere) + Att(x_target, self.p, sphere.pos) )/(tau*tau)

        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        return self.p, self.dp, self.ddp

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
            p[i], dp[i], ddp[i] = self.step(x[i], dt[i], tau[i], self.demo_p[i]) # added target

        return p, dp, ddp

    def reset(self):
        self.p = self.p0.copy()
        self.dp = np.zeros(3)
        self.ddp = np.zeros(3)

    def fit_repulsion(self, positions, ts, tau):
        p = positions
        # Sanity-check input
        if len(p) != len(ts):
            raise RuntimeError("len(p) != len(ts)")
        

    def train(self, positions, ts, tau):
        self.demo_p = positions # added demo_p
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

        print(A.shape)
        print(f.shape)

        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_p = p
        self.train_d_p = d_p
        self.train_dd_p = dd_p
>>>>>>> c961f70e25f731f75ec6caf135d5c1b98b02ef89
