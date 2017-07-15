
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def reArrangeBasis(B, mu, numPoints):

    # Local Variables: B, i, Btemp, numPoints, mure, mu, Bre, mutemp
    # Function calls: reshape, size, reArrangeBasis
    for i in np.arange(1., (matcompat.size(B, 2.))+1):
        Btemp = np.reshape(B[:,int(i)-1], numPoints, 3.).conj().T
        Bre[:,int(i)-1] = Btemp.flatten(1)
        
    mutemp = np.reshape(mu, numPoints, 3.).conj().T
    mure = mutemp.flatten(1)
    return [Bre, mure]