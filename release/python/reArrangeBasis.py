
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def reArrangeBasis(B, mu, numPoints):
    Bre=np.zeros(B.shape)
    # Local Variables: B, i, Btemp, numPoints, mure, mu, Bre, mutemp
    # Function calls: reshape, size, reArrangeBasis
    for i in np.arange(0, B.shape[1]):
        Btemp = B[:,i].reshape(numPoints, 3,order='F').conj().T
        Bre[:,i] = Btemp.flatten(1)
        
    mutemp = mu.reshape(numPoints, 3,order='F').conj().T
    mure = mutemp.flatten(1)
    return Bre, mure