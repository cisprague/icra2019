# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np

class Segment(object):

    '''
    Represents a single trajectory segment, characterised by two boundary
    conditions.
    '''

    def __init__(self, dynamics, s0, sf):

        # set internal dynamics
        self.dynamics = dynamics

        # set bounds
        self.set(s0, sf, t0, tf)

    def set(self, s0, sf):

        ''' Sets the initial and final states. '''

        # states
        self.s0 = np.array(s0, float)
        self.sf = np.array(sf, float)
