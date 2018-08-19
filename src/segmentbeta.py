# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np, matplotlib.pyplot as plt, pygmo as pg
from scipy.integrate import ode

class Segment(object):

    def __init__(self, dynamics):

        # assign dynamics
        self.dynamics = dynamics

        # numerical integrator
        self.integrator = ode(self.eom, jac=self.eom_jac)
        self.integrator.set_integrator('dop853', rtol=1e-8)
        self.integrator.set_solout(self.record)

    def eom(self, time, state):
        return self.dynamics.eom_state(state, self.control(time, state))

    def eom_jac(self, time, state):
        return self.dynamics.eom_state_jac(state, self.control(time, state))

    def set_s0(self, state):
        self.s0       = np.array(state)
        self.integrator.set_initial_value(state)
        self.states   = np.empty((0, len(state)), float)
        self.controls = np.empty((1,0), float)
        self.times    = np.empty((1,0), float)

    def record(self, time, state):
        self.states   = np.vstack((self.states, state))
        self.controls = np.append(self.controls, self.control(time, state))
        self.times    = np.append(self.times, time)

    def set_constraints(self, p0, pf):

        # set constaints
        self.p0 = np.array(p0)
        self.pf = np.array(pf)

    def simulate(self, control, T, verbose=False, method='dop853'):

        # reset
        self.set_s0(self.s0)

        # set controller
        self.control = control

        # simulate
        return self.integrator.integrate(T)

    def plot_traj(self):

        # if axis is not given
        fig, ax = plt.subplots(1)

        # compute pendulum endpoints
        trace = np.vstack((self.states[:,0] + np.sin(self.states[:,2]), np.cos(self.states[:,2]))).T
        alpha = np.linspace(0, 0.5, len(trace))
        for i in range(len(trace)):
            p = np.vstack((np.hstack((self.states[i,0], [0])), trace[i,:]))
            ax.plot(p[:,0], p[:,1], 'k.-', alpha=alpha[i])
        ax.set_aspect('equal')
        ax.set_xlabel(r"x~[m]")
        ax.set_ylabel(r"y~[m]")
        return ax

    def plot(self):

        # number of state variables
        ns = self.states.shape[1]

        # plot
        fig, ax = plt.subplots(ns + 1, sharex=True)

        # labels
        labels = [
            r'$x$ [m]', r'$v$ [m/s]', r'$\theta$ [rad]', r'$\omega$ [rad/s]',
            r'$\lambda_{x}$', r'$\lambda_{v}$', r'$\lambda_{\theta}$', r'$\lambda_{\omega}$'
        ]

        # plot each variable
        for i in range(ns):
            ax[i].plot(self.times, self.states[:,i], 'k-')
            ax[i].set_ylabel(labels[i])
        ax[-1].plot(self.times, self.controls, 'k-')
        ax[-1].set_ylabel(r'$u$')
        ax[-1].set_xlabel(r'$t$ [s]')

        return ax

    def get_nobj(self):
        return 1

    def get_nec(self):
        nec = self.dynamics.sdim
        return nec

    def gradient(self, z):
        return pg.estimate_gradient(self.fitness, z)

    def solve(self):

        # instantiate algorithm
        algo = pg.ipopt()
        algo.set_numeric_option("tol", 1e-5)
        #algo.set_integer_option("max_iter", 100)
        #algo = pg.algorithm(pg.cstrs_self_adaptive(iters=5100, algo=pg.de(100)))
        algo = pg.algorithm(algo)
        algo.set_verbosity(1)

        # instantiate problem
        prob = pg.problem(self)
        pop = pg.population(prob, 1)
        pop = algo.evolve(pop)

        # set tolerences
        prob.c_tol = [1e-4]*self.get_nec()

        '''
        # solve
        while True:
            # create population
            pop = pg.population(prob, 1)
            # evolve
            pop = algo.evolve(pop)
            if prob.feasibility_x(pop.champion_x):
                break
            else:
                print("Trying new guess.")
                continue
        '''

        # return solution
        return pop.champion_x

class Indirect(Segment):

    def __init__(self, dynamics, alpha=0, bound=True):

        # initialise as segment
        Segment.__init__(self, dynamics)

        # default homotopy parameter
        self.alpha = float(alpha)

        # bounded control
        self.bound = bool(bound)

    def eom(self, time, fullstate):
        return self.dynamics.eom_fullstate(fullstate, self.control(time, fullstate))

    def eom_jac(self, time, fullstate):
        return self.dynamics.eom_fullstate_jac(fullstate, self.control(time, fullstate))

    def simulate(self, T, verbose=False):
        return Segment.simulate(self, self.control, T, verbose=verbose)

    def control(self, t, fullstate):
        return self.dynamics.pontryagin(fullstate, self.alpha, bound=self.bound)

    def get_bounds(self):
        lb = [1, *[-10]*self.dynamics.sdim]
        ub = [10, *[10]*self.dynamics.sdim]
        return (lb, ub)

    def get_nec(self):
        return self.dynamics.sdim 

    def fitness(self, z):

        # sanitise
        z = np.array(z)

        # duration
        T = z[0]

        # costates
        l0 = z[1:]

        # set the initial state
        self.set_s0(np.hstack((self.p0, l0)))

        # simulate
        sf = self.simulate(T)

        # ceq
        ceq = np.array([sf[1], sf[2], sf[3]])

        # compute mismatch
        #ceq = self.pf - sf[:self.dynamics.sdim]

        # compute final Hamiltonian
        H = self.dynamics.hamiltonian(sf, self.control(T, sf), self.alpha)

        ceq = np.hstack(([1], ceq, [H]))

        #print(z, ceq)
        return ceq



if __name__ == "__main__":

    # import dynamics
    from dynamics import Dynamics

    # instantiate leg object
    seg = Indirect(Dynamics(), bound=True, alpha=0.99)

    # set boundaries
    seg.set_constraints([0,0,np.pi,0],[0,0,0,0])

    # solve
    sol = seg.solve()

    seg.fitness(sol)
    seg.plot_traj(); seg.plot(); plt.show()

    print(sol)
