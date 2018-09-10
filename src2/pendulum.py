# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import pygmo as pg, numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count
import os
dp = os.path.dirname(os.path.realpath(__file__)) + "/"

def homotopy(traj, damax=0.01, otol=1e-5, iter=200, Tlb=1, Tub=25, lb=100, atol=1e-12, rtol=1e-12):

    a  = 0
    a0 = 0
    g  = None

    x0 = traj[0,1:5]
    xf = traj[-1,1:5]
    dvo = np.hstack((traj[-1,0], traj[0,5:-1]))

    while True:
        print(a)
        dv, s, t, x, u = solve(x0, xf, a, dv=dvo, otol=otol, iter=iter, Tlb=Tlb, Tub=Tub, lb=lb, atol=atol, rtol=rtol)
        if s:
            print("yes")
            ao  = a
            dvo = dv
            if a < 0.9999:
                a   = (a + 1)/2
                if a - ao > damax:
                    a = ao + damax
            elif a == 1:
                break
            else:
                a = 1
        else:
            print("no")
            a = (ao + a)/2

    return np.vstack((t, x.T, u)).T


def random_walks(t, x, alpha, nn, npts, nwalks, dxmax=0.01, otol=1e-5, iter=200, Tlb=1, Tub=25, lb=100, atol=1e-12, rtol=1e-12, fname=None):

    # number of integration nodes
    n = x.shape[1]

    # indicies
    ind = np.linspace(int(n*0.05), int(n*0.8), nn, dtype=int)

    # sample trajectory
    Ts = t[-1] - t[ind]
    xls = x[ind,:]

    # final state
    xf = x[-1,:4]

    # walk arguments
    args = [(T, xl0, xf, alpha, npts, dxmax, otol, iter, Tlb, Tub, lb, atol, rtol) for _ in range(nwalks) for T, xl0 in zip(Ts, xls)]

    # parallel pool
    trajs = Pool(cpu_count()).starmap(random_walk, args)
    trajs =  np.concatenate(trajs)

    if fname is not None:
        np.save(fname, trajs)

    return trajs

def random_walk(T, xl0, xf, alpha, npts, dxmax=0.01, otol=1e-5, iter=200, Tlb=1, Tub=25, lb=100, atol=1e-12, rtol=1e-12):

    # states and costates
    x0 = np.copy(xl0[:4])
    l0 = np.copy(xl0[4:])

    # decision vector
    dvo = np.hstack((T, l0))

    # nominal perturbation size
    dx = dxmax

    # records
    trajs = list()

    # random walk sequence
    i = 0
    while i < npts:
        print("Point {}".format(i))
        xo = np.copy(x0)
        x0 += np.array([5, 2, np.pi, 1], float)*np.random.uniform(-dx, dx, 4)
        dv, feas, t, y, u = solve(x0, xf, alpha, dv=dvo, otol=otol, iter=iter, Tlb=Tlb, Tub=Tub, lb=lb, atol=atol, rtol=rtol)
        if feas:
            traj = np.vstack((t, y.T, u)).T
            trajs.append(traj)
            i += 1
            xo = np.copy(x0)
            dvo = np.copy(dv)
            dx = min(dx*2, dxmax)
        else:
            dx /= 2
            x0 = np.copy(xo)

    return np.array(trajs)

def plot_controls(t, u, ax=None, mark='k-', alpha=1):

    # if axis is provided
    if ax is None:
        fig, ax = plt.subplots(1)

    # plot controls
    ax.plot(t, u, mark, alpha=alpha)

    # equal aspect ratio
    ax.set_aspect('equal')

    return ax

def plot_traj(traj, ax=None, mark='k-', alpha=1):

    # if axis provided
    if ax is None:
        fig, ax = plt.subplots(1)

    # compute endpoints
    x = traj[:,0] + np.sin(traj[:,2])
    y = np.cos(traj[:,2])

    # plot endpoints
    ax.plot(x, y, mark, alpha=alpha)

    # equal aspect ratio
    ax.set_aspect('equal')

    return ax

def solve(x0, xf, alpha, dv=None, otol=1e-5, iter=200, Tlb=1, Tub=25, lb=100, atol=1e-12, rtol=1e-12):

    # initialise dynamics
    dyn = dynamics(x0, xf, alpha, Tlb=Tlb, Tub=Tub, lb=lb, atol=atol, rtol=rtol)

    # optimisation problem
    prob = pg.problem(dyn)
    prob.c_tol = 1e-5

    # algorithm
    algo = pg.ipopt()
    algo.set_numeric_option("acceptable_tol", otol)
    algo.set_integer_option("max_iter", iter)
    algo = pg.algorithm(algo)
    algo.set_verbosity(1)

    # guess
    if dv is None:
        pop = pg.population(prob, 1)
    else:
        pop = pg.population(prob, 0)
        pop.push_back(dv)

    # solve
    dv = algo.evolve(pop).champion_x

    # feasibility
    feas = prob.feasibility_x(dv)

    T = dv[0]
    l0 = dv[1:]
    t, y, s, f = dyn.propagate(T, l0)
    return dv, feas, t, y.T, dyn.pmp(y, alpha)


class dynamics(object):

    def __init__(self, x0, xf, alpha, Tlb=1, Tub=25, lb=1, atol=1e-12, rtol=1e-12):

        # boundary constraints
        self.x0 = x0
        self.xf = xf

        # homotopy parameter
        self.alpha = alpha

        # time bounds
        self.Tlb = Tlb
        self.Tub = Tub

        # costate magnitude
        self.lb = lb

        # tolerances
        self.atol = atol
        self.rtol = rtol


    @staticmethod
    def dxdt(x, u):

        # state
        x, v, theta, omega = x

        # state dynamics
        return np.array([v, u, omega, -u*np.cos(theta) + np.sin(theta)], float)

    @staticmethod
    def dlxdt(xl, u):

        # fullstate
        x, v, theta, omega, lx, lv, ltheta, lomega = xl

        # common subexpression elimination
        e0 = np.sin(theta)
        e1 = np.cos(theta)

        # fullstate dynamics
        return np.array([
            v, u, omega, -u*e1 + e0,
            0, -lx, -lomega*(u*e0 + e1), -ltheta
        ], float)

    @staticmethod
    def pmp(xl, alpha):

        # extract state and costate
        x, v, theta, omega, lx, lv, ltheta, lomega = xl

        # Pontryagin's minimum principle
        if alpha == 1:
            return -np.sign(lomega*np.cos(theta)-lv)
        else:
            return np.clip((-lomega*np.cos(theta) + lv)/(2*(alpha - 1)), -1, 1)

    def propagate(self, T, l0):

        # integrate
        if callable(l0):
            pass
        else:

            sol = solve_ivp(
                lambda t, xl: self.dlxdt(xl, self.pmp(xl, self.alpha)),
                (0, T), np.hstack((self.x0, l0)),
                method='RK45', atol=self.atol, rtol=self.rtol
            )

        return sol.t, sol.y, sol.success, sol.sol

    def fitness(self, dv):

        # duration and costates
        T = dv[0]
        l0 = dv[1:]

        # simulate
        t, x, s, f = self.propagate(T, l0)

        # mismatch
        ec = self.xf - x[:4,-1]

        # fitness vector
        return np.hstack(([1], ec))

    def get_bounds(self):
        lb = [self.Tlb] + [-self.lb]*4
        ub = [self.Tub] + [self.lb]*4
        return lb, ub

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 4

    def gradient(self, dv):
        return pg.estimate_gradient(self.fitness, dv)

if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
