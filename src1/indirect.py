# Christopher Iliffe Sprague
# sprague@kth.se

from segment import Segment
import numpy as np, matplotlib.pyplot as plt, pygmo as pg
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

class Indirect(Segment):

    def __init__(self, x0, xf):
        Segment.__init__(self, x0, xf)

    def encode(self, T, l0):
        # concatenate duration and costates
        return np.hstack(([T], l0))

    def decode(self, dv):
        # duration
        T = dv[0]
        # costates
        l0 = dv[1:]
        # return tuple
        return T, l0

    def controls(self, xl, alpha):
        return np.apply_along_axis(lambda xl: self.pmp(xl, alpha), 1, xl)

    def propagate(self, T, l0, alpha, atol=1e-10, rtol=1e-10, controls=False):

        sol = solve_ivp(
            lambda t, xl: self.eom_fullstate(xl, self.pmp(xl, alpha)),
            (0, T),
            np.hstack((self.x0, l0)),
            method='RK45',
            atol=atol,
            rtol=rtol,
            jac=lambda t, xl: self.eom_fullstate_jac(xl, self.pmp(xl, alpha))
        )

        # states and time
        tl, xl = sol.t, sol.y.T

        # controls
        if controls:
            ul = np.apply_along_axis(lambda xl: self.pmp(xl, alpha), 1, xl)
            return tl, xl, ul
        else:
            return tl, xl

    def fitness(self, dv):

        # extract duration and costates
        T, l0 = self.decode(dv)

        # simulate
        t, xl = self.propagate(T, l0, self.alpha, atol=self.atol, rtol=self.rtol)

        # mistmatch
        ec = self.xf - xl[-1,:self.xdim]

        # fitness vector
        fit = np.hstack(([1], ec))

        # return fitness vector
        return fit

    def get_bounds(self):
        lb = [self.Tlb] + [-self.lb]*self.xdim
        ub = [self.Tub] + [self.lb]*self.xdim
        return lb, ub

    def get_nobj(self):
        return 1

    def get_nec(self):
        return self.xdim

    def gradient(self, dv):
        return pg.estimate_gradient(self.fitness, dv)

    def solve(self, alpha, Tlb=1, Tub=30, lb=1, atol=1e-10, rtol=1e-10, otol=1e-5, iter=200, dv=None, verbose=False, auto=False):

        # set homotopy parameter
        self.alpha = alpha

        # set time bounds
        self.Tlb = Tlb
        self.Tub = Tub

        # set costate magnitude tolerances
        self.lb = lb

        # set tolerances
        self.atol = atol
        self.rtol = rtol

        # problem
        prob = pg.problem(self)
        prob.c_tol = otol

        # algorithm
        algo = pg.ipopt()
        algo.set_numeric_option("tol", otol)
        algo.set_integer_option("max_iter", iter)
        algo = pg.algorithm(algo)
        algo.set_verbosity(1)

        # supplied population
        if dv is not None:
            if verbose: print("Testing supplied decision vector: {}".format(dv))
            pop = pg.population(prob, 0)
            pop.push_back(dv)
            pop = algo.evolve(pop)
            feas = prob.feasibility_x(pop.champion_x)
            if verbose:
                print("Supplied decision vector was {}.".format("succesfull" if feas else "unsuccesfull"))
            return pop.champion_x, feas

        # random population
        else:
            pop = pg.population(prob, 1)
            if verbose: print("Trying random decision vector {}.".format(pop.champion_x))
            pop = algo.evolve(pop)
            if verbose: print("Optimised decision vector now {}".format(pop.champion_x))
            feas = prob.feasibility_x(pop.champion_x)
            if verbose:
                print("Supplied decision vector was {}.".format("succesfull" if feas else "unsuccesfull"))
            return pop.champion_x, feas

    def homotopy(self, atol=1e-10, rtol=1e-10, otol=1e-5, lb=1, iter0=200, iter=50, alpha0=0, verbose=False, savef=None):

        dva = list()

        # try to load prexisting record
        try:
            if verbose: print("Trying to load prexisting homotopy.")
            dva   = np.load(savef + ".npy")
            #dva = dva.reshape((-1, self.xdim + 1))
            dvo   = dva[-1,:-1]
            alpha = dva[-1,-1]
            if verbose:
                print("Found prexisting homotopy!")
                print("The last line is {}".format(dva[-1,:]))

            if alpha == float(1) or alpha == int(1):
                if verbose:
                    print("This homotopy record is already optimised.")
                return None
        except:
            if verbose:
                print("No prexisting homotopy found at {}, creating one.".format(savef))
            dva   = np.empty(shape=(0,self.xdim+1), dtype=float)
            dvo   = None
            alpha = alpha0

        # cut off homotopy parameter
        alphatol = 0.9999
        # optimal homotopy parameter
        alphao = 0

        # initial solution
        if verbose: print("Solving for initial trajectory...")
        i = 0
        while True:
            dvo, success = self.solve(alpha, dv=dvo, atol=atol, rtol=rtol, otol=otol, lb=lb, iter=iter0, Tlb=2, Tub=24)
            if success:
                if verbose: print("Found initial trajectory.")
                break
            else:
                i += 1
                if verbose:
                    print("Trying new decision vector.")
                    print("{} unsuccesfull tries.".format(i))

            if i > 100:
                return None

        if verbose: print("Found DV = {}".format(dvo))

        # homotopy sequence
        if verbose: print("Initiating homotopy sequence...")
        i=0
        while True:

            dv, success = self.solve(alpha, dv=dvo, iter=iter, atol=atol, rtol=rtol, Tlb=2, Tub=24)

            if success:

                # record optimal paramters
                if verbose: print("Success at {}, DV = {}".format(alpha, dv))
                alphao = alpha
                dvo    = dv
                dva.append(np.hstack((dv, [alpha])))
                if savef is not None:
                    np.save(savef, np.array(dva))

                # increase homotopy parameter
                if alpha < alphatol:
                    alpha = (1+alpha)/2
                    if verbose: print("Increasing to {}".format(alpha))

                # full homotopy parameter value
                elif alpha >= alphatol:
                    if verbose: print("Solving for bang-bang...")
                    alpha = 1
                elif alpha == 1:
                    if verbose: print("Found bang-bang...")
                    return np.array(dvl)

            # shrink homotopy parameter
            else:
                if alpha == 1:
                    i += 1
                    if verbose: print("{} unsuccesfull TOC tries.".format(i))
                if i > 10:
                    return None
                alpha = (alpha+alphao)/2
                if verbose: print("Decreasing to {}".format(alpha))

    def random_walk(self, T, xl0, alpha, dx=0.01, atol=1e-10, rtol=1e-10, iter=10, verbose=False, npts=20):

        # decision vector and initial state list
        dvl, x0l = list(), list()

        # initial state
        x0 = xl0[:self.xdim]

        # initial costate
        l0 = xl0[self.xdim:]

        # decision vector
        dvo = self.encode(T, l0)

        xold = np.copy(self.x0)

        # random walk sequence
        i, j = 0, 0
        if verbose: print("Beginning random walk from {}".format(xl0))
        while True:
            # original initial state
            xo = np.copy(self.x0)
            print(self.x0)
            self.x0 += (self.xlb - self.xub)*np.random.uniform(-dx, dx, self.xdim)
            print(self.x0)
            dv, feas = self.solve(dv=dvo, alpha=alpha, iter=iter)

            # if succesfull
            if feas:
                i += 1
                j = 0
                dvl.append(np.copy(dv))
                x0l.append(np.copy(self.x0))
                xo = np.copy(self.x0)
                dx *= 2
                dvo = np.copy(dv)
                if verbose:
                    print("Feasible! dx now {}".format(dx))
                if i == npts:
                    break

            # if failed
            else:
                j += 1
                dx /= 2
                self.x0 = np.copy(xo)
                if verbose:
                    print("Not feasible! dx now {}".format(dx))
                if j == 20:
                    break

        self.x0 = np.copy(xold)
        return np.array(dvl), np.array(x0l)

    def random_walks(self, dv, alpha, dx=0.01, atol=1e-10, rtol=1e-10, iter=10, verbose=False, nn=20, npts=20):

        # decode decision vector
        T, l0 = self.decode(dv)

        # propagate trajectory
        tl, xl = self.propagate(T, l0, alpha)
        tf = tl[-1]*1

        # number of trajectory nodes
        n = len(tl)

        # indicies
        ind = np.linspace(0, n-int(0.1*n), nn, dtype=int)

        # optimal trajectory samples
        tl, xl = tl[ind], xl[ind,:]

        # compute durations
        Tl = np.array([tf - tl[i] for i in range(nn)])

        # parallel args
        args = [(T, xl0, alpha, dx, atol, rtol, iter, verbose, npts) for T, xl0 in zip(Tl, xl)]

        # parallel pool
        p = Pool(cpu_count())

        return p.map(self.random_walk_args, args)

    def random_walk_args(self, args):
        return self.random_walk(*args)

if __name__ == "__main__":

    seg = Indirect([0,0,np.pi,0],[0,0,0,0],0,3)
    dv  = np.hstack(([3], np.random.uniform(-1,1,seg.xdim)))
    seg.random_walks(dv, 1)
