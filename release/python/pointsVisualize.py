
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def pointsVisualize(vals, skel, varargin):

    # Local Variables: gt, missI, missJ, color, h1, colors, connect, marker, linewidth, skel, varargin, type, handle, strs, missing, I, showlines, J, M, vals, textsize, inds, i, texton, person, indices
    # Function calls: set, pointsVisualize, plot3, text, line, prism, logical, ind2sub, length, ones, isempty, process_options, num2str, skelConnectionMatrix, find, size
    #%function pointsVisualize(vals,skel)
    return 