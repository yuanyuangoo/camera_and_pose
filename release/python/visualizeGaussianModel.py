
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def visualizeGaussianModel(vals, skel, varargin):

    # Local Variables: color, gaussianMan, colors, connect, zv, xv, person, znew, newXYZ, skel, varargin, type, P2, C, P1, handle, y, I, J, M, ynew, P, S, U, xnew, vals, yv, strs, i, indices, x, z
    # Function calls: set, eye, ellipsoid, visualizeGaussianModel, reshape, surf2patch, int2str, length, sqrt, patch, prism, isempty, plot3, process_options, ind2sub, skelConnectionMatrix, null, find, norm, size
    
    return [gaussianMan]