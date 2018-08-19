# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np

class Direct(object):

    def __init__(self, segment, N):

        # assign segment
        self.segment = segment

        # number of segments
        self.N = int(N)

        # initialise grids
        self.set(*self.linear())


    def set(self, times, states, controls):

        self.times    = np.array(times, float)
        self.states   = np.array(states, float)
        self.controls = np.array(controls, float)

    def linear(self):

        # linearly interpolate with N points
        times    = np.linspace(self.segment.t0, self.segment.tf, self.N)
        states   = np.vstack((
            np.linspace(self.segment.s0[i], self.segment.sf[i], self.N)
            for i in range(self.segment.dynamics.sdim)
        )).T
        controls = np.random.random((self.N, self.segment.dynamics.cdim))
        return times, states, controls

    def mismatch(self):

        # equality constraint array
        ceq = np.empty((self.N + 1, self.segment.dynamics.sdim), float)

        for i in range(self.N-1):
            ceq[i] = self.quadrature(
                self.states[i], self.states[i+1],
                self.controls[i], self.controls[i+1],
                self.times[i+1] - self.times[i]
            ) - self.states[i+1]

        # terminal constraints
        ceq[self.N-1] = np.array(self.states[0]-self.segment.s0)
        ceq[self.N]   = np.array(self.states[-1]-self.segment.sf)

        # return constraints vector
        return ceq


class Hermite_Simpson(Direct):

    def __init__(self, dynamics, N):

        # initialise as direct segment
        Direct.__init__(self, dynamics, N)

    def quadrature(self, s0, s1, u0, u1, h0):

        '''
        Returns definite integral approximation between two states and times.
        '''

        f0  = self.segment.dynamics.eom_state(s0, u0)
        f1  = self.segment.dynamics.eom_state(s1, u1)
        s05 = (s0 + s1)/2 + h0*(f0 + f1)/8
        u05 = (u0 + u1)/2
        f05 = self.segment.dynamics.eom_state(s05, u0)
        return s0 + h0*(f0 + 4*f05 + f1)/6

class Problem(object):

    def __init__(self, transcription):

        # assign direct transcription
        self.transcription = transcription

    def fitness(self, dv):

        '''
        Fitness vector:
        [t0,s0,u0,...,tf,sf,uf]
        '''

        # reshape decision vector
        dv = dv.reshape((
            self.transcription.N,
            1 +
            self.transcription.segment.dynamics.sdim +
            self.transcription.segment.dynamics.cdim
        ))

        # extract times
        i, j = 0, 0
        times = dv[:,i].flatten()

        # extract state
        i = 1; j += i + self.transcription.segment.dynamics.sdim
        states = dv[:,i:j]

        # extract controls
        i = j; j +=  i + self.transcription.segment.dynamics.cdim
        controls = dv[:,i:j]

        # set transcription
        self.transcription.set(times, states, controls)

        # get collocation mismatch
        ec = self.transcription.mismatch().flatten()

        # time mismatch
        ic = np.array([times[i]-times[i+1] for i in range(self.transcription.N-1)], float)

        # compute objective
        obj = np.trapz(
            self.transcription.segment.dynamics.lagrangian(states, controls),
            times
        )

    def get_bounds(self):

        lb = [
            0,
            *self.transcription.segment.dynamics.slb,
            *self.transcription.segment.dynamics.clb
        ]*self.transcription.N
        ub = [
            0,
            *self.transcription.segment.dynamics.sub,
            *self.transcription.segment.dynamics.cub
        ]*self.transcription.N

        return lb, ub

    def get_nobj(self):
        return 1

    def get_nec(self):
        return self.transcription.N*self.transcription.segment.dynamics.sdim

    def get_nix(self):
        return self.transcription.N

    def gradient(self, z):
        pg.estimate_gradient(self.fitness, z)










if __name__ == "__main__":

    from dynamics import Dynamics
    from segment import Segment
    seg = Segment(Dynamics(), [0,0,0,0], [1,0,0,0], 0, 10)
    disc = Hermite_Simpson(seg, 10)
    t, s, u = disc.linear()
    u = u.reshape(disc.N, 1)
    z = np.hstack((t.reshape(disc.N, 1), s, u))
    prob = Problem(disc)
    print(prob.get_bounds())
