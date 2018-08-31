# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from indirect import Indirect
import numpy as np
from multiprocessing import Pool, cpu_count

# list of boundary pairs
conds = [

    # down to up
    ([0,0,np.pi,0],[-1,0,0,0]),
    ([0,0,np.pi,0],[0,0,0,0]),
    ([0,0,np.pi,0],[1,0,0,0]),
    ([-1,0,np.pi,0],[-1,0,0,0]),
    ([-1,0,np.pi,0],[0,0,0,0]),
    ([-1,0,np.pi,0],[1,0,0,0]),
    ([1,0,np.pi,0],[-1,0,0,0]),
    ([1,0,np.pi,0],[0,0,0,0]),
    ([1,0,np.pi,0],[1,0,0,0]),

    # up to down
    ([0,0,0,0],[-1,0,np.pi,0]),
    ([0,0,0,0],[0,0,np.pi,0]),
    ([0,0,0,0],[1,0,np.pi,0]),
    ([-1,0,0,0],[-1,0,np.pi,0]),
    ([-1,0,0,0],[0,0,np.pi,0]),
    ([-1,0,0,0],[1,0,np.pi,0]),
    ([1,0,0,0],[-1,0,np.pi,0]),
    ([1,0,0,0],[0,0,np.pi,0]),
    ([1,0,0,0],[1,0,np.pi,0]),

    # down to down
    ([0,0,np.pi,0],[-1,0,np.pi,0]),
    ([0,0,np.pi,0],[1,0,np.pi,0]),
    ([-1,0,np.pi,0],[0,0,np.pi,0]),
    ([-1,0,np.pi,0],[1,0,np.pi,0]),
    ([1,0,np.pi,0],[-1,0,np.pi,0]),
    ([1,0,np.pi,0],[0,0,np.pi,0]),

    # up to up
    ([0,0,0,0],[-1,0,0,0]),
    ([0,0,0,0],[1,0,0,0]),
    ([-1,0,0,0],[0,0,0,0]),
    ([-1,0,0,0],[1,0,0,0]),
    ([1,0,0,0],[-1,0,0,0]),
    ([1,0,0,0],[0,0,0,0]),

]

def homotopy(pair):

    # verbosity
    print("Performing homotopy on pair: {}".format(pair))

    # instantiate segment
    seg = Indirect(pair[0], pair[1])

    # file name
    savef = "../data/homotopy/" + str(pair)

    # perform homotopy
    dvs = seg.homotopy(verbose=True, iter=1000, iter0=400, atol=1e-12, rtol=1e-12, lb=1, savef=savef)

if __name__ == "__main__":


    # number of CPUs
    n = cpu_count()
    print("Executing with {} CPU cores.".format(n))

    # parallel pool
    p = Pool(n)

    p.map(homotopy, conds)
