# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from indirect import Indirect
from direct import Direct
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import permutations

# nominal configurations
xfl = [[x,0,theta,0] for x in [0,1] for theta in [0,np.pi]]

x0fl = [([0, 0, 0, 0], [0, 0, 3.141592653589793, 0]),
 ([0, 0, 0, 0], [1, 0, 0, 0]),
 ([0, 0, 0, 0], [1, 0, 3.141592653589793, 0]),
 ([0, 0, 3.141592653589793, 0], [0, 0, 0, 0]),
 ([0, 0, 3.141592653589793, 0], [1, 0, 0, 0]),
 ([0, 0, 3.141592653589793, 0], [1, 0, 3.141592653589793, 0]),
 ([1, 0, 0, 0], [0, 0, 0, 0]),
 ([1, 0, 0, 0], [0, 0, 3.141592653589793, 0]),
 ([1, 0, 3.141592653589793, 0], [0, 0, 0, 0]),
 ([1, 0, 3.141592653589793, 0], [0, 0, 3.141592653589793, 0])
]

# Step 1: generate nominal trajectories for each behaviour
def nominal_energy(pair):

    # verbosity
    print("Finding minimum time energy optimal dv for: {}.".format(pair))

    # instantiate segment
    seg = Indirect(*pair)

    # result destination
    fname = "../data/nominal_energy/" + str(pair) + ".npy"
    print(fname)

    # check if record already exists
    try:

        # get trajectory
        traj = np.load(fname)

        # nominal duration
        To = traj[-1,0] - traj[0,0]

        # verbosity
        print("Found prexisting trajectory.")

    # if we didn't find a prexisting trajectory
    except:

        print("Didn't find prexisting trajectory")

        # use nominal upper time bound
        To = 20

    # failure counter
    i = 0

    # optimise repeatedly
    while i < 5:

        # solve trajectory
        dv, feas = seg.solve(0, Tub=To+0.1, iter=300, atol=1e-12, rtol=1e-12)

        # duration
        T = dv[0]

        # verbosity
        print("Found DV {} with feasibility {}".format(T, feas))

        # if solution feasible and faster
        if T < To and feas:

            # verbosity
            print("Best time now {}.".format(T))

            # set optimal time
            To = T

            # compute trajectory with optimal decision vector
            tl, xl, ul = seg.propagate(*seg.decode(dv), 0, controls=True)

            # assemble full data
            traj = np.vstack((tl, xl.T, ul)).T

            # save trajectory
            np.save(fname, traj)

        # if feasible and not faster
        elif feas:
            i += 1
            print("Not better")

        # if infeasbile
        else:
            print("Infeasible")

def homotopy(pair):

    # verbosity
    print("Performing homotopy on pair: {}".format(pair))

    # instantiate segment
    seg = Indirect(*pair)

    # load nominal trajectory
    nom = np.load("../data/nominal_energy/" + str(pair) + ".npy")

    # construct decision vector
    T = nom[-1,0] - nom[0,0]
    l0 = nom[0,1+seg.xdim:1+2*seg.xdim]
    dv = np.hstack(([T], l0))

    # file name to save homotopy
    fname = "../data/nominal_homotopy/" + str(pair)

    # perform homotopy
    dvs = seg.homotopy(dv=dv, verbose=True, iter=200, atol=1e-12, rtol=1e-12, lb=1000, fname=fname)

def random_walks_energy(pair):

    # verbosity
    print("Performing random walks on pair: {}".format(pair))

    # instantiate segment
    seg = Indirect(*pair)

    # file name
    fname = "../data/walks_energy/" + str(pair) + ".npy"

    # load nominal trajectory
    traj = np.load("../data/nominal_energy/" + str(pair) + ".npy")
    tl = traj[:,0]
    xl = traj[:,1:1+2*seg.xdim]
    ul = traj[:,-1]

    # do random walks
    res = seg.random_walks(tl, xl, 0, verbose=True, npts=20, nwalks=2, nn=2, dxmax=0.01)

    # save result
    np.save(fname, res)


if __name__ == "__main__":

    print(x0fl)


    # number of CPUs
    #n = cpu_count()
    #print("Executing with {} CPU cores.".format(n))

    # parallel pool
    #p = Pool(n)
    #p.map(homotopy, x0fl)

    #p.map(random_walks_energy, x0fl)
    [random_walks_energy(x0f) for x0f in x0fl]
