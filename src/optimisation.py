# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from indirect import Indirect
from direct import Direct
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import permutations

# list of all possible boundary pairs
conds = [[x,0,theta,0] for theta in [0, np.pi] for x in [-1,0,1]]
conds = list(permutations(conds, 2))

def nominal_energy(pair):

    # verbosity
    print("Finding minimum time energy optimal dv for: {}.".format(pair))

    # instantiate segment
    seg = Indirect(*pair)

    # result destination
    fname = "../data/nominal_energy/" + str(pair) + ".npy"

    # check if record already exists
    try:
        dv = np.load(fname)
        To = dv[0]
        print("Found prexisting trajectory.")
    except:
        # minimum time recorded
        To = 30

    # failure counter
    i = 0

    # optimise repeatedly
    while i < 5:

        # solve trajectory
        dv, feas = seg.solve(0.1, Tub=To+1, iter=300)

        # duration
        T = dv[0]

        # verbosity
        print("Found DV {} with feasibility {}".format(T, feas))

        # if solution feasible and faster
        if T < To and feas:
            To = T
            dvo = dv
            print("Best")
            np.save(fname, dvo)

        # if feasible and not faster
        elif feas:
            i += 1
            print("Not better")

        # if infeasbile
        else:
            print("Infeasible")


# homotopy for each condition
def homotopy(pair):

    # verbosity
    print("Performing homotopy on pair: {}".format(pair))

    # instantiate segment
    seg = Indirect(*pair)

    # file name
    fname = "../data/homotopy/" + str(pair)

    # perform homotopy
    dvs = seg.homotopy(verbose=True, iter=500, iter0=500, atol=1e-12, rtol=1e-12, lb=1, fname=fname)


if __name__ == "__main__":


    # number of CPUs
    n = cpu_count()
    print("Executing with {} CPU cores.".format(n))

    # parallel pool
    p = Pool(n)

    p.map(nominal_energy, conds)
