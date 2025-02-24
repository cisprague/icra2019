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
            method='LSODA',
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

    def solve(self, alpha, Tlb=1, Tub=30, lb=1, atol=1e-10, rtol=1e-10, otol=1e-5, iter=200, dv=None, verbose=False):

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
        algo.set_numeric_option("acceptable_tol", otol)
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

    def homotopy(self, damax=0.1, dv=None, atol=1e-12, rtol=1e-12, otol=1e-4, lb=1, iter0=200, iter=100, verbose=False, fname=None):

        # try to load prexisting array
        try:

            # load array - breaks if file doesn't work
            dvah = np.load(fname + '.npy')
            # last optimal decision vector
            dvo   = dvah[-1,:-1]
            # last optimal homotopy parameter
            alpha = dvah[-1,-1]

            # if succesfully loaded
            if verbose:
                print("Found prexisting trajectory.")
                print("The last line is {}".format(dvah[-1,:]))

            # if bang bang control is already achieved
            if alpha == 1:
                if verbose:
                    print("Homotopy record already optimised.")
                # nothing to be done
                return None

        # no prexisting array found
        except:
            if verbose:
                print("Prexisting array not found.")
            # new homotopy record list
            dvah = np.empty(shape=(0,self.xdim+2), dtype=float)
            # optimal decision vector not known yet
            dvo = None if dv is None else dv
            print(dvo)
            # homotopy parameter of zero
            alpha = 0


        # homotopy parameter at which to assume bang-bang
        alphatol = 0.9999
        # best homotopy parameter so far
        alphao = 0


        # find initial solution or polish prexisting
        if verbose: print("Solving initial trajectory...")
        # try 100 times
        i = 0
        while i < 100:

            # solve prescribed trajectory 5 times
            if i < 5:
                dvo, success = self.solve(alpha, dv=dvo, atol=atol, rtol=rtol, otol=otol, lb=lb, iter=iter0, Tlb=2, Tub=25)

            # solve with random DV
            else:
                dvo, success = self.solve(alpha, dv=None, atol=atol, rtol=rtol, otol=otol, lb=lb, iter=iter0, Tlb=2, Tub=25)

            # break if it was succesfull
            if success:
                if verbose:
                    print("Found initial DV = {}".format(dvo))
                break

            # increment the failure counter if the solution failed
            else:
                i += 1
                if verbose:
                    print("Trying new decision vector.")
                    print("{} unsuccesfull tries.".format(i))

            # quite if over 100 tries
            if i > 100:
                return None

        # homotopy sequence
        if verbose: print("Initiating homotopy sequence...")
        i=0
        while True:

            # solve trajectory
            dv, success = self.solve(alpha, dv=dvo, iter=iter, atol=atol, rtol=rtol, Tlb=2, Tub=30)

            # if solution succesfull
            if success:

                # record optimal paramters
                if verbose: print("Success at {}, DV = {}".format(alpha, dv))

                # best decision vector and homotopy parameter
                alphao = alpha
                dvo    = np.copy(dv)

                # record decision vector and homotopy parameter
                dvah = np.vstack((dvah, np.hstack((dvo, [alphao]))))

                # save if file name given
                if fname is not None:
                    np.save(fname + '.npy', dvah)

                # increase homotopy parameter
                if alpha < alphatol:
                    alphanew = (1+alpha)/2
                    alphadiff = alphanew - alpha
                    if alphadiff > damax:
                        alpha += damax
                    else:
                        alpha = alphanew
                    if verbose: print("Increasing to {}".format(alpha))

                # full homotopy parameter value
                elif alpha >= alphatol and alpha < 1:
                    if verbose: print("Solving for bang-bang...")
                    alpha = 1

                # if full bang bang
                elif alpha == 1:
                    if verbose: print("Found bang-bang...")
                    return dvah

            # shrink homotopy parameter
            else:

                # unsuccesfull bang-bang try
                if alpha == 1:
                    i += 1
                    if verbose:
                        print("{} unsuccesfull TOC tries.".format(i))

                # 10 unsuccesfull bang bang tries
                if i == 2:
                    return dvah

                # decrease homotopy parameter
                alpha = (alpha+alphao)/2
                if verbose:
                    print("Decreasing to {}".format(alpha))

    def sample_traj(self, tln, xln, nn=20):

        # number of integration nodes
        n = len(tln)

        # indicies
        ind = np.linspace(int(n*0.1), int(n*0.8), nn, dtype=int)

        # sample times and trajectory
        tls = tln[ind]
        xls = xln[ind,:]

        # nominal duration
        T = tln[-1]

        # durations for each sample
        Tls = T - tls

        # return durations and fullstates
        return Tls, xls

    def random_walk(self, T, xl0, alpha, npts, dxmax=0.01 , iter=100, verbose=True, atol=1e-10, rtol=1e-10):

        # save original initial state
        x0 = np.copy(self.x0)

        # set initial state
        self.x0 = xl0[:self.xdim]

        # initial costate
        l0 = xl0[self.xdim:]

        # decision vector
        dvo = self.encode(T, l0)

        # list of optimal trajectories
        trajectories = list()

        # set nominal perturbation size
        dx = dxmax

        # random walk sequence
        i, j = 0, 0
        if verbose: print("Beginning random walk from {}".format(xl0))
        while i < npts:

            # original initial state
            xo = np.copy(self.x0)
            self.x0 += (self.xub - self.xlb)*np.random.uniform(-dx, dx, self.xdim)
            print(dvo)
            dv, feas = self.solve(dv=dvo, alpha=alpha, iter=iter, lb=1000)

            # if succesfull
            if feas:

                # increment succeses
                i += 1
                # reset failures
                j = 0

                # get trajectory
                tl, xl, ul = self.propagate(*self.decode(dv), alpha, controls=True)

                # consolidate
                traj = np.vstack((tl, xl.T, ul)).T
                trajectories.append(traj)

                # accept new state and decision vector
                xo = np.copy(self.x0)
                dvo = np.copy(dv)

                # double state perturbation
                dx = min(dx*2, dxmax)

                if verbose:
                    print("Feasible! dx now {}".format(dx))
                    print(i)

            # if failed
            else:
                # increment failures
                j += 1
                # halve perturbation size
                dx /= 2
                # reset initial state
                self.x0 = np.copy(xo)
                if verbose:
                    print("Not feasible! dx now {}".format(dx))
                if j == 20:
                    break

        self.x0 = x0
        return np.array(trajectories)

    def random_walks(self, tln, xln, alpha, dxmax=0.001, atol=1e-10, rtol=1e-10, iter=100, verbose=False, npts=20, nwalks=5, nn=5, fname=None):

        # divide trajectory
        Tls, xls = self.sample_traj(tln, xln, nn=nn)

        # random walk processes
        args = [(T, xl0, alpha, npts, 0.01, 20) for _ in range(nwalks) for T, xl0 in zip(Tls, xls)]

        # number of CPUs
        n = cpu_count()
        print("Executing with {} CPU cores.".format(n))

        # parallel pool
        p = Pool(n)

        # results
        res = p.starmap(self.random_walk, args)
        res = np.concatenate(res)

        if fname is not None:
            np.save(fname, res)

        return res

    def plot_homotopy(self, dvah, ax=None):

        # if axis is given
        if ax is None:
            fig, ax = plt.subplots(1)

        # number of trajectories in homotopy sequence
        n = len(dvah)

        # iterate through trajectories
        for i in range(n):

            # decision vector
            dv = dvah[i,:-1]

            # homotopy parameter
            alpha = dvah[i,-1]

            # decode decision vector
            T, l0 = self.decode(dv)

            # propagate trajectory
            tl, xl, ul = self.propagate(T, l0, alpha, controls=True)

            # compute endpoints
            x = xl[:,0] + np.sin(xl[:,2])
            y = np.cos(xl[:,2])

            # plot trajectory
            ax.plot(x, y, "k-", alpha=0.2)
            ax.set_aspect('equal')
        return ax

if __name__ == "__main__":

    seg = Indirect([0,0,0,0],[0,0,0,0])
    tl, xl = seg.propagate(4,np.random.uniform(-1,1,seg.xdim), 0)
    seg.sample_traj(tl, xl)
