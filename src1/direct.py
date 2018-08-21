# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from segment import Segment
import numpy as np, pygmo as pg
from scipy.interpolate import spline

class Direct(Segment):

    def __init__(self, x0, xf, N):
        Segment.__init__(self, x0, xf)
        self.N = N

    def get_bounds(self):
        lb = [0] + ([*self.xlb] + [self.ulb])*self.N
        ub = [10] + ([*self.xub] + [self.uub])*self.N
        return lb, ub

    def encode(self, T, states, controls):

        # states and controls
        sc = np.hstack((states, controls.reshape((self.N, 1))))

        # return decision vector
        return np.hstack(([T], sc.flatten()))

    def decode(self, dv):

        # duration
        T = dv[0]

        # time grid
        times, h = np.linspace(0, T, self.N, retstep=True)

        # states and controls
        sc = dv[1:].reshape((self.N, self.xdim + self.udim))

        # states
        states = sc[:, :self.xdim]

        # controls
        controls = sc[:, -1]

        return times, h, states, controls

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
            J += h*(u0**2 + 4*u05**2 + u1**2)/6

            # mismatch
            ec[k] = h*(f0 + 4*f05 + f1)/6 + x0 - x1

        # terminal constraints
        ec[-2] = self.x0 - states[0]
        ec[-1] = self.xf - states[-1]

        # return fitness vector
        fit = np.hstack(([J], ec.flatten()))
        return fit

    def get_nobj(self):
        return 1

    def get_nec(self):
        return self.xdim*(self.N+1)

    def gradient(self, dv):
        return pg.estimate_gradient(self.fitness, dv)

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
