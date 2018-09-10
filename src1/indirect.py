# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def homotopy(x0, xf, damax=0.1):

    a  = 0
    ao = 0
    g  = None
    rec = list()

    while a <= 0.95:
        print(a)
        x, y, s, f = solve(x0, xf, a, guess=g)
        if s:
            ao = a
            g = (x, y)
            rec.append((x, y))
            a = (a + 1)/2
            if a - ao > damax:
                a = ao + damax
        else:
            a = (ao + a)/2

    return rec

def solve(x0, xf, alpha, guess=None):

    # initialise dynamics
    dyn = dynamics(x0, xf, alpha)

    # time and state guess
    t, s = dyn.linear_guess(1, 50) if guess is None else guess

    # duration
    p = [1.]

    # solve tpbvp
    res = solve_bvp(dyn.fun, dyn.bc, t, s, p, tol=1e-10, verbose=2, max_nodes=10000)

    return res

def plot_traj(states, ax=None):

    # create figure
    if ax is None:
        fig, ax = plt.subplots(1)

    # endpoints
    x = states[0,:] + np.sin(states[2,:])
    y = np.cos(states[2,:])

    # plot trajectory
    ax.plot(x, y, "k-")

    # equal aspect ratio
    ax.set_aspect("equal")

    # y lims
    ax.set_ylim(-1,1)

    return ax


class dynamics(object):

    def __init__(self, x0, xf, alpha):

        # boundary constraints
        self.x0 = x0
        self.xf = xf

        # homotopy parameter
        self.alpha = alpha

    def fun(self, t, fs, p):

        # states
        x, v, theta, omega, lx, lv, ltheta, lomega = fs

        # Pontryagin's minimum principle
        if self.alpha == 1:
            u = -np.sign(lomega*np.cos(theta)-lv)
        else:
            u = (-lomega*np.cos(theta) + lv)/(2*(self.alpha - 1))
            u = np.clip(u, -1, 1)

        # common subexpression elimination
        e0 = np.sin(theta)
        e1 = np.cos(theta)

        # state dynamics
        dx     = v
        dv     = u
        dtheta = omega
        domega = -u*e1 + e0

        # costate dynamics
        dlx     = np.zeros(fs.shape[1])
        dlv     = -lx
        dltheta = -lomega*(u*e0 + e1)
        dlomega = -ltheta

        # return fullstate transiton
        return np.vstack((dx, dv, dtheta, domega, dlx, dlv, dltheta, dlomega))*p[0]

    def bc(self, fs0, fsf, p):

        # initial state mismatch
        ec0 = fs0[:4] - self.x0

        # final state mismatch
        ec1 = fsf[:4] - self.xf

        # return constraint vector
        return np.hstack((ec0, ec1, [0]))

    def linear_guess(self, T, nn):

        # times
        t = np.linspace(0, T, nn)

        # states
        s = np.vstack((
            *(np.linspace(l,u,nn) for l,u in zip(self.x0, self.xf)),
            np.random.uniform(-1,1,(4,nn))
        ))

        return t, s
