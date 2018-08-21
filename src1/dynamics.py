# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np

class Dynamics(object):

    def __init__ (self):

        # state and control dimensions
        self.xdim = 4
        self.udim = 1

        # state bounds
        self.xlb = [-10, -10, 0, -10]
        self.xub = [10, 10, 2*np.pi, 10]

        # control bounds
        self.ulb = -2
        self.uub = 2

    # equations of motion
    @staticmethod
    def eom(x, u):
        return np.array([x[1], u, x[3], np.sin(x[2]) - u*np.cos(x[2])], float)
