# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from segment import Segment
import numpy as np, matplotlib.pyplot as plt, pygmo as pg
from scipy.interpolate import CubicSpline
from scipy.integrate import simps

class Direct(Segment):

    def __init__(self, x0, xf):
        Segment.__init__(self, x0, xf)

    def encode(self, T, states, controls):

        # states and controls
        sc = np.hstack((states, controls.reshape((-1, 1))))

        # return decision vector
        return np.hstack(([T], sc.flatten()))

    def decode(self, dv):

        # duration
        T = dv[0]

        # states and controls
        sc = dv[1:].reshape((-1, self.xdim + self.udim))

        # number of nodes
        N = len(sc)

        # time grid
        times, h = np.linspace(0, T, N, retstep=True)

        # states
        states = sc[:, :self.xdim]

        # controls
        controls = sc[:, -1]

        return times, h, states, controls

    def interpolate(self, dv, N):

        # decode decision vector
        times, h, states, controls = self.decode(dv)

        # new times
        timesn = np.linspace(times[0], times[-1], N)

        # new states
        statesn = CubicSpline(times, states, bc_type="natural")(timesn)
        statesn = np.vstack((
            np.clip(statesn[:,i], self.xlb[i], self.xub[i]) for i in range(self.xdim)
        )).T

        # new controls
        controlsn = CubicSpline(times, controls, bc_type="natural")(timesn)
        controlsn = np.clip(controlsn, -self.ub, self.ub)

        # encode decision vector
        return self.encode(times[-1], statesn, controlsn)

    def linear_guess(self, N, T=None):

        # states
        states = np.vstack((
            np.linspace(l, u, N) for l, u in zip(self.x0, self.xf)
        )).T

        # controls
        controls = np.random.uniform(-self.ub, self.ub, N)

        # duration
        if T is None:
            T = np.random.uniform(self.Tlb, self.Tub)
        else:
            T = T

        # encode decision vector
        return self.encode(T, states, controls)

    def get_bounds(self):
        lb = [self.Tlb] + ([*self.xlb] + [-self.ub])*self.N
        ub = [self.Tub] + ([*self.xub] + [self.ub])*self.N
        return lb, ub

    def fitness(self, dv):

        # decode decision vector
        times, h, states, controls = self.decode(dv)

        # object function
        J = 0

        # equality constraints
        ec = np.empty((self.N+1, self.xdim), float)

        # hermite simpson collocation
        for k in range(self.N-1):

            # initial, middle, and terminal controls
            u0  = controls[k]
            u1  = controls[k+1]
            u05 = (u0 + u1)/2

            # initial and terminal states
            x0 = states[k]
            x1 = states[k+1]

            # initial and terminal dynamics
            f0 = self.eom(x0, u0)
            f1 = self.eom(x1, u1)

            # middle state
            x05 = (x0 + x1)/2 + h*(f0-f1)/8

            # middle dynamics
            f05 = self.eom(x05, u05)

            # objective
            if self.obj == 'energy':
                J += h*(u0**2 + 4*u05**2 + u1**2)/6

            # mismatch
            ec[k] = h*(f0 + 4*f05 + f1)/6 + x0 - x1

        # terminal constraints
        ec[-2] = self.x0 - states[0]
        ec[-1] = self.xf - states[-1]

        # duration optimisation
        if self.obj == 'time':
            J = dv[0]

        # return fitness vector
        fit = np.hstack(([J], ec.flatten()))
        return fit

    def get_nobj(self):
        return 1

    def get_nec(self):
        return self.xdim*(self.N+1)

    def gradient(self, dv):
        return pg.estimate_gradient(self.fitness, dv)

    def solve(self, inp, Tlb=2, Tub=30, obj='time'):

        self.Tlb = Tlb
        self.Tub = Tub

        # if guess
        if isinstance(inp, int):
            self.N = inp
            guess = False
        elif inp is not None:
            times, h, states, controls = self.decode(inp)
            self.N = len(states)
            guess = True

        if obj == 'energy':
            self.obj = 'energy'
        elif obj == 'time':
            self.obj = 'time'
        else:
            raise ValueError("Object must be 'energy' or 'time'.")

        # problem
        prob = pg.problem(self)

        # population
        if guess:
            pop = pg.population(prob, 0)
            pop.push_back(inp)
        else:
            pop = pg.population(prob, 1)

        # algorithm
        algo = pg.ipopt()
        algo.set_numeric_option("acceptable_tol", 1e-5)
        algo.set_integer_option("max_iter", 500)
        algo = pg.algorithm(algo)
        algo.set_verbosity(1)

        # evolve population
        pop = algo.evolve(pop)

        # return decision vector
        return pop.champion_x

    def plot_traj(self, dv, ax=None):

        # decode decision vector
        times, h, states, controls = self.decode(dv)

        # create figure
        if ax is None:
            fig, ax = plt.subplots(1)

        # compute endpoints
        x = states[:,0] + np.sin(states[:,2])
        y = np.cos(states[:,2])

        # plot points
        ax.plot(x, y, "k-")

        # plot arms
        for i in range(len(states)):
            ax.plot([x[i], states[i, 0]], [y[i], 0], "k.-", alpha=0.1)

        # equal aspect ratio
        ax.set_aspect("equal")
        
        # return the axis
        return ax

    def plot_timeline(self, dv):

        # decode decision vector
        times, h, states, controls = self.decode(dv)

        # create figure
        fig, ax = plt.subplots(self.xdim + 1, sharex=True)

        # plot states
        for i in range(self.xdim):
            ax[i].plot(times, states[:,i], "k-")

        # plot control
        ax[-1].plot(times, controls, "k-")

        return ax










if __name__ == "__main__":

    # instantiate segment
    seg = Direct([0,0,np.pi,0],[1,0,0,0],50)

    # duration guess
    T = 3

    # linear state guess
    states = np.vstack((
        np.linspace(l, u, seg.N) for l,u in zip(seg.x0, seg.xf)
    )).T

    # random controls
    controls = np.random.random(seg.N)
    print(controls)

    # PyGMO
    """
    prob = pg.problem(seg)
    pop  = pg.population(prob, 1)
    algo = pg.algorithm(pg.ipopt())
    algo.set_verbosity(1)
    pop = algo.evolve(pop)
    """
