from __future__ import division, print_function

import numpy as np


class CanonicalSystem(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.step_vectorized = np.vectorize(self.step, otypes=[float])
        self.reset()

    def step(self, dt, tau):
        """
        Solve the canonical system at next time step t+dt.

        Parameters
        ----------
        dt : float
            Time step.
        tau : float
            Temporal scaling factor.
        """
        # TODO: Implement the canonical system differential equation, given that you know the values of the following
        # variables:
        # self.x, self.alpha, dt, tau
        self.x += dt * ( -self.alpha * self.x ) / tau # since x_t+1 = x_t + dt*(xdot) and tau*xdot = -alpha*x
        return self.x

    def rollout(self, t, tau):
        """
        Solve the canonical system.

        Parameters
        ----------
        t : array_like
            Time points for which to evaluate the integral.
        tau : array_like
            Temporal scaling factor (scalar constant or same length as t).
        """
        self.reset()
        return self.step_vectorized(np.gradient(t), tau)

    def reset(self):
        self.x = 1.0
