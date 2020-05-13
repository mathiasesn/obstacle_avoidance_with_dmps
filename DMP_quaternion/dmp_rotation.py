from __future__ import division, print_function

import numpy as np
from pyquaternion import Quaternion
import copy
#from scipy.spatial.transform import Rotation as R

from canonical_system import CanonicalSystem


class RotationDMP():
    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None, cs=None):
        self.n_bfs = n_bfs # Number of basis functions
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        # Canonical system object
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
        self.q0 = Quaternion(np.zeros(4))
        self.gq = Quaternion(np.zeros(4))

        self.reset()

    def step(self, x, dt, tau):
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

        f = fp(x)
        f_alt = np.array([0, f[0], f[1], f[2]])
        
        #self.eta[0] = 0
        
        self.eta_dot = (self.alpha*(self.beta*2*Quaternion.log( self.gq * self.q.conjugate ) - self.eta)+f_alt)

        #self.eta_dot[0] = 0

        self.eta += self.eta_dot * dt / tau

        self.q = Quaternion.exp((dt/2)*self.eta/tau)*self.q
        return self.q, self.eta/tau, self.eta_dot

    def rollout(self, ts, tau):
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)
        
        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)
        q = [Quaternion([0,0,0,0]) for _ in range(n_steps)]
        dq = [Quaternion([0,0,0,0]) for _ in range(n_steps)]
        ddq = [Quaternion([0,0,0,0]) for _ in range(n_steps)]

        for i in range(n_steps):
            q[i], dq[i], ddq[i] = self.step(x[i], dt[i], tau[i])
        
        return q, dq, ddq

    def reset(self):
        #self.q = copy.copy(self.q0)
        self.q = Quaternion(self.q0)
        self.eta = Quaternion(np.zeros(4))
        self.eta_dot = Quaternion(np.zeros(4))

    def train(self, rotations, ts, tau):
        q = rotations

        # Sanity-check input
        if len(q) != len(ts):
            raise RuntimeError("len(p) != len(ts)")

        # Initial- and goal positions
        self.q0 = Quaternion(q[0])
        self.gq = Quaternion(q[-1])

        # Differential time vector
        dt = np.gradient(ts)[:,np.newaxis]

        # Scaling factor
        # ***** Change to be the diagonal of the vector elements of 2*log(g*q.conj)
        error = 2*Quaternion.log(self.gq*self.q0.conjugate)
        self.Dp = np.diag(error.vector)
        Dp_inv = np.linalg.inv(self.Dp)

        # Desired velocities and accelerations
        # ***** Change to be the values of 2*log(g*q.conj) or some form of velocity and acceleration of the training data.
        d_q = [Quaternion([0,0,0,0])]
        dd_q = [Quaternion([0,0,0,0])]
        for i in range(len(q)-1):
            d_q.append( 2*Quaternion.log(q[i+1]*q[i].conjugate) / dt[i] )
        for i in range(len(d_q)-1):
            dd_q.append( (d_q[i+1] - d_q[i]) / dt[i] )
        dd_q.append(Quaternion([0, 0, 0, 0]))


        # Integrate canonical system
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return xj * psi / psi.sum()

        def forcing(j):
            vel = 2*Quaternion.log(self.gq * q[j].conjugate)
            return Dp_inv.dot(tau**2 * dd_q[j].vector - self.alpha * (self.beta * (vel.vector) - tau * d_q[j].vector))

        A = np.stack(features(xj) for xj in x)
        f = np.stack(forcing(j) for j in range(len(ts)))
        
        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_q = q
        self.train_d_q = d_q
        self.train_dd_q = dd_q
        
