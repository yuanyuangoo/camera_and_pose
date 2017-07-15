
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def projectIntoAffineCam(X, K, R, t, S, skel, varargin):

    # Local Variables: strs, J, handle, i, indices, taff, I, numPts, K, Raff, M, s, skel, S, xy, t, varargin, X, R, viz, connect
    # Function calls: isempty, axis, set, figure, plot, projectIntoAffineCam, int2str, line, length, ones, zeros, process_options, ind2sub, skelConnectionMatrix, repmat, text, find, size
    #%xy=projectIntoAffineCam(X, K,R,t,S,skel )
    [viz, s] = process_options(varargin, 'viz', 0., 'scale', 1.)
    #%% Do projection
    numPts = matcompat.size(X, 1.)
    #%Create Affine camera
    Raff = np.array(np.vstack((np.hstack((R[0,:])), np.hstack((R[1,:])), np.hstack((np.zeros(1., 3.))))))
    taff = np.dot(K, np.array(np.vstack((np.hstack((t[0])), np.hstack((t[1])), np.hstack((1.))))))
    #% TODO: z-dim should be scaling
    M = np.dot(K, Raff)
    #%Project into affine camera
    xy = np.dot(np.dot(S, M[0:2.,:]), X.conj().T)+matcompat.repmat(taff[0:2.], np.array(np.hstack((1., numPts))))
    xy[2,:] = np.ones(1., matcompat.size(xy, 2.))
    #%% Plot projection
    if viz:
        connect = skelConnectionMatrix(skel)
        indices = nonzero(connect)
        [I, J] = ind2sub(matcompat.size(connect), indices)
        for i in np.arange(1., (numPts)+1):
            strs.cell[int(i)-1] = cellarray(np.hstack((int2str(i))))
            
        if not isempty(skel):
            plt.figure
            handle[0] = plt.plot(xy[0,:], xy[1,:], 'x')
            plt.hold(on)
            for i in np.arange(1., (matcompat.size(xy, 2.))+1):
                plt.text(xy[0,int(i)-1], xy[1,int(i)-1], strs.cell[int(i)-1])
                
            plt.axis(equal)
            plt.hold(off)
            if K[0,2] > 0.:
                if K[1,2] > 0.:
                    plt.axis(np.array(np.hstack((0., 2.*K[0,2], 0., 2.*K[1,2]))))
                
                
            
            
            plt.axis(ij)
            #% make sure the left is on the left.
            set(handle[0], 'markersize', 20.)
            plt.hold(on)
            plt.grid(on)
            for i in np.arange(1., (length(indices))+1):
                handle[int((i+1.))-1] = line(np.array(np.hstack((xy[0,int(I[int(i)-1])-1], xy[0,int(J[int(i)-1])-1]))), np.array(np.hstack((xy[1,int(I[int(i)-1])-1], xy[1,int(J[int(i)-1])-1]))))
                set(handle[int((i+1.))-1], 'linewidth', 2.)
                
            plt.hold(off)
        
        
    
    
    return [xy]