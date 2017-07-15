
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def alignToCamera(X, R, t, Rnew, tnew):

    # Local Variables: tnew, R, Rnew, t, Xnew, X
    # Function calls: length, repmat, alignToCamera
    #% [Xnew]=alignToCamera(X,R,t,Rnew,tnew)
    #%
    #% R, t - Current Camera
    #% Rnew, tnew - Camera to be aligned to.
    #%
    #%
    Xnew = np.dot(R, X.conj().T)+matcompat.repmat(t, 1., length(X))
    Xnew = np.dot(Rnew, Xnew)+matcompat.repmat(tnew, 1., length(X))
    Xnew = Xnew.conj().T
    return [Xnew]