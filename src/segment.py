# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np
from dynamics import Dynamics

class Segment(Dynamics):

    def __init__(self, x0, xf):
        Dynamics.__init__(self)
        self.x0 = np.array(x0, float)
        self.xf = np.array(xf, float)
