# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from indirect import Indirect
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import combinations

# list of all possible boundary pairs
conds = [[x,0,theta,0] for theta in [0, np.pi] for x in [-1,0,1]]
conds = list(combinations(conds, 2))

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

    p.map(homotopy, conds)
