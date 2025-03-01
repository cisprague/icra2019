# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class Dynamics(object):

    def __init__ (self):

        # state and control dimensions
        self.xdim = 4
        self.udim = 1

        # state bounds
        self.xlb = np.array([-5, -5, -2*np.pi, -3])
        self.xub = np.array([5, 5, 2*np.pi, 3])

        # control bounds
        self.ub = 1

    # equations of motion
    def eom(self, x, u):

        # extract state
        x, v, theta, omega = x

        # return state transition
        return np.array([v, u, omega, np.sin(theta) - u*np.cos(theta)], float)

    def eom_fullstate(self, xl, u):

        # extract fullstate
        x, v, theta, omega, lx, lv, ltheta, lomega = xl

        # common subexpression elimination
        e0 = np.sin(theta)
        e1 = np.cos(theta)

        # fullstate transition
        return np.array([
            v,
            u,
            omega,
            e0 - u*e1,
            0,
            -lx,
            -lomega*(u*e0 + e1),
            -ltheta
        ], float)

    def eom_fullstate_jac(self, xl, u):

        e0 = np.cos(xl[2])
        e1 = np.sin(xl[2])
        e2 = u*e1

        return np.array([
            [0, 1,                           0, 0,  0, 0,  0,        0],
            [0, 0,                           0, 0,  0, 0,  0,        0],
            [0, 0,                           0, 1,  0, 0,  0,        0],
            [0, 0,                     e0 + e2, 0,  0, 0,  0,        0],
            [0, 0,                           0, 0,  0, 0,  0,        0],
            [0, 0,                           0, 0, -1, 0,  0,        0],
            [0, 0,          -xl[7]*(u*e0 - e1), 0,  0, 0,  0, -e0 - e2],
            [0, 0,                           0, 0,  0, 0, -1,        0]
        ], float)

    def pmp(self, xl, alpha):

        # extract fullstate
        x, v, theta, omega, lx, lv, ltheta, lomega = xl

        # time optimal control
        if alpha == 1:

            return -self.ub*np.sign(lomega*np.cos(theta)-lv)

        else:
            # unbounded optimal control
            u = (lv-lomega*np.cos(theta))/(2*(alpha-1))
            return min(max(u, -self.ub), self.ub)

    def hamiltonian(self, xl, u):

        # extract fullstate
        x, v, theta, omega, lx, lv, ltheta, lomega = xl

        return lomega*(-u*np.cos(theta) + np.sin(theta)) + \
        ltheta*omega + lv*u + lx*v + u**2

    @staticmethod
    def plot_traj(states, interp=False, ax=None, arm=True, pts=True):

        if ax is None:
            fig, ax = plt.subplots(1)

        # endpoint positions
        x = states[:,0] + np.sin(states[:,2])
        y = np.cos(states[:,2])

        # number of nodes
        N = len(states)

        # plot arm
        if arm:
            for i in range(N):
                ax.plot([x[i], states[i, 0]], [y[i], 0], "k.-", alpha=0.1)

        if pts:
            ax.plot(x, y, "k-")

        # plot interpolant if desired
        if interp:
            states = CubicSpline(np.linspace(0,1,N), np.vstack((x, y)).T, bc_type="natural")(np.linspace(0,1,1000))
            ax.plot(states[:,0], states[:,1], "k--")

        ax.set_aspect('equal')

        return ax

    @staticmethod
    def plot_timeline(times, states, controls, interp=False, ax=None, mark="k-"):

        if ax is None:
            fig, ax = plt.subplots(self.xdim+self.udim, sharex=True)

        for i in range(self.xdim):
            ax[i].plot(times, states[:,i], mark)

        ax[-1].plot(times, controls, mark)

        if interp:
            timesn = np.linspace(times[0], times[-1], 1000)
            statesn = CubicSpline(times, states, bc_type="natural")(timesn)
            controlsn = CubicSpline(times, controls, bc_type="natural")(timesn)
            for i in range(self.xdim):
                ax[i].plot(timesn, statesn[:, i], "k--")
            ax[-1].plot(timesn, controlsn, "k--")

        return ax
