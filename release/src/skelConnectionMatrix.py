
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def skelConnectionMatrix(skel):

    # Local Variables: i, skel, connection, j
    # Function calls: skelConnectionMatrix, zeros, length
    #% SKELCONNECTIONMATRIX Compute the connection matrix for the structure.
    #%
    #%	Description:
    #%
    #%	CONNECTION = SKELCONNECTIONMATRIX(SKEL) computes the connection
    #%	matrix for the structure. Returns a matrix which has zeros at all
    #%	entries except those that are connected in the skeleton.
    #%	 Returns:
    #%	  CONNECTION - connectivity matrix.
    #%	 Arguments:
    #%	  SKEL - the skeleton for which the connectivity is required.
    #%	
    #%
    #%	See also
    #%	SKELVISUALISE, SKELMODIFY
    #%	Copyright (c) 2006 Neil D. Lawrence
    #% 	skelConnectionMatrix.m CVS version 1.2
    #% 	skelConnectionMatrix.m SVN version 42
    connection = np.zeros(length((skel.tree)))
    for i in np.arange(1., (length((skel.tree)))+1):
        for j in np.arange(1., (length((skel.tree[int(i)-1].children)))+1):
            connection[int(i)-1,int((skel.tree[int(i)-1].children[int(j)-1]))-1] = 1.
            
        
    return [connection]