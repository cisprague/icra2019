# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np

class Segment(object):

    '''
    Represents a single trajectory segment, characterised by two boundary
    conditions.
    '''

    def __init__(self, dynamics, s0, sf, t0, tf):

        # set internal dynamics
        self.dynamics = dynamics

        # set bounds
        self.set(s0, sf, t0, tf)

    def set(self, s0, sf, t0, tf):

        ''' Sets the initial and final states and times. '''

        # states
        self.s0 = np.array(s0, float)
        self.sf = np.array(sf, float)

        # times
        self.t0 = np.array(t0, float)
        self.tf = np.array(tf, float)
