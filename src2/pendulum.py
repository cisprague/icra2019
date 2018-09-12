# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import pygmo as pg, numpy as np, matplotlib.pyplot as plt, torch
from scipy.integrate import solve_ivp, RK45
from multiprocessing import Pool, cpu_count
import os
from scipy.optimize import fsolve
dp = os.path.dirname(os.path.realpath(__file__)) + "/"
from scipy.interpolate import CubicSpline

def propagate_controlled(x0, xf, T, controller):
    t, x, u = dynamics(x0, xf, 0).propagate_controlled(T, controller)
    return t, x, u

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

def plot_traj(traj, n=500, ax=None, mark='k-', alpha=1, arm=False):

    # plot interpolant
    traj = CubicSpline(
        np.linspace(0,1,traj.shape[0]),
        traj,
        bc_type="natural"
    )(np.linspace(0,1,n))

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

    # plot arm
    if arm:
        for i in range(traj.shape[0]):
            ax.plot([x[i], traj[i, 0]], [y[i], 0], "k.-", alpha=0.1)

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
    return dv, feas, t, y, dyn.pmp(y.T, alpha)


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
            return np.clip((-lomega*np.cos(theta) + lv)/(2*(alpha - 1)), -2, 2)

    def propagate_controlled(self, T, controller):

        integrator = RK45(
            lambda t, x: self.dxdt(x, controller(x)),
            0,
            self.x0,
            T,
            0.01,
            rtol=1e-12,
            atol=1e-12
        )

        tl = [integrator.t]
        xl = [integrator.y]
        ul = []

        while integrator.status is 'running':
            integrator.step()
            tl.append(integrator.t)
            xl.append(integrator.y)
            ul.append(controller.control(integrator.y))

        ul.append(ul[-1])

        return np.array(tl), np.array(xl), np.array(ul)

    def propagate(self, T, l0):
        sol = solve_ivp(
            lambda t, xl: self.dlxdt(xl, self.pmp(xl, self.alpha)),
            (0, T), np.hstack((self.x0, l0)),
            method='RK45', atol=self.atol, rtol=self.rtol
        )
        return sol.t, sol.y.T, sol.success, sol.sol

    def fitness(self, dv):

        # duration and costates
        T = dv[0]
        l0 = dv[1:]

        # simulate
        t, x, s, f = self.propagate(T, l0)

        # mismatch
        ec = self.xf - x[-1,:4]
        ec[0] = x[-1, 4]

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

class srinivasan(object):

    def __init__(self, kx=0.1, kv=0.5, kt=10, kw=10, um=2):
        self.tr = 0
        self.kx = kx
        self.kv = kv
        self.kt = kt
        self.kw = kw
        self.ul = list()
        self.trl = [self.tr]
        self.um = um

    def gen_tt(self, tr):
        xm = np.pi**2*(np.pi + self.um)
        vm = np.pi*self.um
        eq = lambda tt: self.kt*(tt[0]-tr) + np.sin(tt[0]) + self.kt*np.arctan(self.kx*xm + self.kv*vm)
        res = fsolve(eq, [0])
        return res[0]

    def gen_tr(self, t, tr):
        tt = self.gen_tt(tr)
        dt = t - tr
        if -np.pi <= dt and dt < -tt:
            return tr - 2*np.pi
        elif -tt <= dt and dt <= tt:
            return tr
        elif tt < dt and dt <= np.pi:
            return tr + 2*np.pi
        else:
            return tr

    def gen_u(self, x, v, t, w, tr):
        u = (self.kt*(t-tr) + self.kw*w + np.sin(t) + self.kt*np.arctan(self.kx*x + self.kv*v))/np.cos(t)
        u = np.clip(u, -self.um, self.um)
        return u

    def __call__(self, state): # NOTE: for intermediate integration steps
        x, v, t, w = state
        u = self.gen_u(x, v, t, w, self.tr)
        return u

    def control(self, state): # NOTE: for succesfull integration steps
        x, v, t, w = state
        u = self(state)
        self.tr = self.gen_tr(t, self.tr)
        self.ul.append(u)
        self.trl.append(self.tr)
        return u

class mlp(torch.nn.Sequential):

    def __init__(self, shape, name="MLP"):

        # architecture
        self.shape = shape

        # number of inputs
        self.ni = self.shape[0]

        # number of outputs
        self.no = self.shape[-1]

        # number of layers
        self.nl = len(self.shape)

        # loss data
        self.ltrn, self.ltst = list(), list()
        self.dropout = False

        # name
        self.name = name

        # operations
        #self.ops = [torch.nn.BatchNorm2d(self.shape[0])]
        self.ops = list()

        # apply operations
        for i in range(self.nl - 1):

            # linear layer
            self.ops.append(torch.nn.Linear(self.shape[i], self.shape[i + 1]))

            # batch normalisation
            #self.ops.append(torch.nn.BatchNorm2d(self.shape[i + 1]))


            # if penultimate layer
            if i == self.nl - 2:

                # output between 0 and 1
                self.ops.append(torch.nn.Tanh())
                pass

            # if hidden layer
            else:

                # activation
                self.ops.append(torch.nn.LeakyReLU())
                #self.ops.append(torch.nn.AlphaDropout(p=0.2))
                pass

        # initialise neural network
        torch.nn.Sequential.__init__(self, *self.ops)
        self.double()

    @staticmethod
    def format(x):
        if isinstance(x, np.ndarray):
            x = torch.autograd.Variable(torch.from_numpy(x)).double()
        elif isinstance(x, torch.DoubleTensor):
            x = torch.autograd.Variable(x).double()
        elif isinstance(x, torch.autograd.Variable):
            x = x.double()
        else:
            raise TypeError("Unable to format input.")
        return x

    def forward(self, x):

        # format input
        x = self.format(x)

        # apply operations
        for op in self.ops:
            x = op(x)

        # return transformation
        return x

    def predict(self, x):

        # format
        x = self.format(x).data.numpy()

        # predict
        return self.forward(x).data.numpy()

    def train(self, idat, odat, epo=50, batches=10, lr=1e-4, ptst=0.1):

        # numpy
        idat, odat = [dat.numpy() for dat in [idat, odat]]

        # number of testing samples
        ntst = int(ptst*idat.shape[0])

        # training data
        itrn, otrn = [dat[ntst:] for dat in [idat, odat]]

        # testing data
        itst, otst = [dat[:ntst] for dat in [idat, odat]]

        # delete original data
        del idat; del odat

        # batch data
        itrn, otrn, itst, otst = [np.array_split(dat, batches) for dat in [itrn, otrn, itst, otst]]

        # tensor
        itrn, otrn, itst, otst = [[torch.from_numpy(d) for d in dat] for dat in [itrn, otrn, itst, otst]]

        # variables
        itrn, otrn, itst, otst = [[torch.autograd.Variable(d) for d in dat] for dat in [itrn, otrn, itst, otst]]

        # optimiser
        opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        #opt = torch.optim.Adadelta(self.parameters(), lr=lr)

        # loss function
        lf = torch.nn.MSELoss()

        # for each episode
        for t in range(epo):

            # average episode loss
            ltrne, ltste = list(), list()

            # zero gradients
            opt.zero_grad()

            # for each batch
            for itrnb, otrnb, itstb, otstb in zip(itrn, otrn, itst, otst):

                # training loss
                ltrn = lf(self.forward(itrnb), otrnb)

                # testing loss
                ltst = lf(self.forward(itstb), otstb)

                # record loss
                ltrne.append(float(ltrn.data[0]))
                ltste.append(float(ltst.data[0]))

                # progress
                print(self.name, t, ltrne[-1], ltste[-1])

                # backpropagate training error
                ltrn.backward()

            # update weights
            opt.step()

            self.ltrn.append(np.average(ltrne))
            self.ltst.append(np.average(ltste))

        return self

class data(object):

    def __init__(self, data, cin, cout, shuffle=True):

        # cast as numpy array
        data = np.array(data)

        # shuffle rows
        if shuffle:
            np.random.shuffle(data)

        # number of samples
        self.n = data.shape[0]

        # cast to torch
        self.i = torch.from_numpy(data[:, cin]).double()
        self.o = torch.from_numpy(data[:, cout]).double()

class mlp_controller(object):

    def __init__(self, mlp):
        self.mlp = mlp

    def __call__(self, state):
        return self.mlp.predict(state)[0]

    def control(self, state):
        return self(state)




if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
