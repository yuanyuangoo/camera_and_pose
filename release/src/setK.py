
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def setK(x, y, f):

    # Local Variables: y, x, K, f
    # Function calls: eye, setK
    K = np.eye(3.)
    K[0,0] = np.dot(f, x)
    K[1,1] = np.dot(f, x)
    K[0,2] = x/2.
    K[1,2] = y/2.
    return [K]