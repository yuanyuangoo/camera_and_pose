import numpy as np

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
    t=skel['tree']
    connection = np.zeros((len(t),len(t)))
    for i in np.arange(0, len(t)):
        if isinstance(t[i].children,int):
            tmp=1
            connection[i,(t[i].children)-1] = 1
        else:
            tmp=len(t[i].children)
            for j in np.arange(0, tmp):
                connection[i,(t[i].children)[j]-1] = 1
    return connection