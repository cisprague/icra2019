# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

if __name__ == "__main__":

    import sys; sys.path.append("../src/")
    from dynamics import Dynamics
    from segment import Indirect
    import pygmo as pg, matplotlib.pyplot as plt, numpy as np

    # instantiate segment
    seg = Indirect(Dynamics())

    seg.set(np.hstack((np.zeros(seg.dynamics.sdim), np.random.random(seg.dynamics.sdim))))
    seg.set_constraints([0,0,0,0], [1,0,0,0])

    # instantiate algorithm
    algo = pg.ipopt()
    algo.set_numeric_option("tol", 1e-8)
    algo.set_integer_option("max_iter", 20)
    algo = pg.algorithm(algo)
    algo.set_verbosity(1)

    # instantiate problem
    prob = pg.problem(seg)

    # set tolerences
    prob.c_tol = [1e-4]*seg.dynamics.sdim

    i = 0
    while True:
        i += 1
        pop = pg.population(prob, 1)
        pop = algo.evolve(pop)
        if prob.feasibility_x(pop.champion_x):
            break
        elif i == 50:
            break
        else:
            continue

    seg.fitness(pop.champion_x)
    seg.plot()
    seg.plot_traj()
    plt.show()
