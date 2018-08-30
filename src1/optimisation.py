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
    seg = Indirect(pair[0], pair[1], 1, 30)

    # file name
    savef = "../data/homotopy/" + str(pair)

    # perform homotopy
    dvs = seg.homotopy(verbose=True, iter=200, iter0=200, atol=1e-12, rtol=1e-12, lb=1, savef=savef)

def main1(pair):

    # instantiate segment
    seg = Indirect(*pair, 1, 30)

    # solve
    while True:
        dv, success = seg.solve(alpha=0, atol=1e-12, rtol=1e-12, otol=1e-5, lb=1, iter=200)
        if success:
            print("Found solution {}".format(dv))
            break
        else:
            continue

    # save solution
    np.save("../data/qc/" + str(pair), dv)

if __name__ == "__main__":


    # number of CPUs
    n = cpu_count()
    print("Executing with {} CPU cores.".format(n))

    # parallel pool
    p = Pool(n)

    p.map(homotopy, conds)
