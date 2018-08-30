# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np

class Dynamics(object):

    def __init__(self):

        # state and control dimensions
        self.xdim = 4
        self.udim = 1

        # state bounds
        self.xlb = [-5, -3, -np.pi*2, -1]
        self.xub = [5, 3, np.pi*2, 1]

        # control bounds
        self.ulb = -1
        self.uub = 1


    def eom_state(self, state, control):

        # extract states variables
        y, dy, theta, dtheta = state

        # extract control input
        v = control

        # return equations of motion
        return np.array([
            dy,
            v,
            dtheta,
            np.sin(theta) - v*np.cos(theta)
        ], float)

    def eom_state_jac(self, state, control):

        # extract states variables
        y, dy, theta, dtheta = state

        # extract control input
        v = control

        # return the Jacobian matrix
        return np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, v*np.sin(theta) + np.cos(theta), 0]
        ], float)

    def eom_fullstate(self, state, control):

        # extract fullstate variables
        y, dy, theta, dtheta, ly, ldy, ltheta, ldtheta = state

        # extract control input
        v = control

        # common subexpression elimination
        x0 = np.sin(theta)
        x1 = np.cos(theta)

        # return state transition matrix
        return np.array([
            dy,
            v,
            dtheta,
            -v*x1 + x0,
            0,
            -ly,
            -ldtheta*(v*x0 + x1),
            -ltheta
        ], float)

    def eom_fullstate_jac(self, fullstate, control):

        # extract fullstate variables
        y, dy, theta, dtheta, ly, ldy, ltheta, ldtheta = fullstate

        # extract control input
        v = control

        # common subexpression elimination
        x0 = np.cos(theta)
        x1 = np.sin(theta)
        x2 = v*x1

        # return fullstate Jacobian
        return np.array([
            [0, 1,                                 0, 0,  0, 0,  0,        0],
            [0, 0,                                 0, 0,  0, 0,  0,        0],
            [0, 0,                                 0, 1,  0, 0,  0,        0],
            [0, 0,                           x0 + x2, 0,  0, 0,  0,        0],
            [0, 0,                                 0, 0,  0, 0,  0,        0],
            [0, 0,                                 0, 0, -1, 0,  0,        0],
            [0, 0,              ldtheta*(-v*x0 + x1), 0,  0, 0,  0, -x0 - x2],
            [0, 0,                                 0, 0,  0, 0, -1,        0]
        ], float)

    def pontryagin(self, fullstate, alpha, bound=False):

        # extract fullstate variables
        y, dy, theta, dtheta, ly, ldy, ltheta, ldtheta = fullstate

        # compute optimal control
        uo = (alpha - ldtheta*np.cos(theta) + ldy)/(2*(alpha - 1))

        if bound:
            return min(max(uo, -1), 1)
        else:
            return uo

    def hamiltonian(self, fullstate, control, alpha):

        # extract fullstate variables
        y, dy, theta, dtheta, ly, ldy, ltheta, ldtheta = fullstate

        # extract control input
        v = control

        # return Hamiltonian
        return alpha*v + dtheta*ltheta + dy*ly + \
            ldtheta*(-v*np.cos(theta) + np.sin(theta)) + ldy*v + \
            v**2*(-alpha + 1)
