
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def plot2Dskeleton(xy, skel, type, varargin):

    # Local Variables: gt, I, strs, skel, i, hSkelText, J, M, texton, varargin, colors, xy, connect, numPts, hSkel, indices, linewidth, type, hparent
    # Function calls: gca, set, plot2Dskeleton, text, int2str, line, length, prism, ind2sub, process_options, skelConnectionMatrix, find, size
    #%PLOT2DSKELETON  Plots projection of skeleton
    #%    plot2Dskeleton(xy,skel)
    #%
    #%   Description:
    #%
    #%
    #%   Inputs:
    #%
    #%
    #%   Outputs:
    #%
    #%
    #%   Example:
    #%     plot2Dskeleton
    #%
    #%   See also
    #% Author: Varun Ramakrishna
    #% Created: Jan 31, 2011
    xy = xy.conj().T
    [texton, linewidth, gt, hparent] = process_options(varargin, 'texton', 1., 'linewidth', 2., 'gt', 0., 'Parent', plt.gca)
    numPts = matcompat.size(xy, 2.)
    for i in np.arange(1., (numPts)+1):
        strs.cell[int(i)-1] = cellarray(np.hstack((int2str(i))))
        
    connect = skelConnectionMatrix(skel)
    indices = nonzero(connect)
    [I, J] = ind2sub(matcompat.size(connect), indices)
    #% gca; hold on;
    #%figure
    #% hold on;
    #% plot(xy(1,:),xy(2,:),'o','LineWidth',linewidth,'MarkerSize',5);
    colors = plt.prism(6.)
    if type == 2.:
        M
        M
        M
        for i in np.arange(1., (matcompat.size(xy, 2.))+1):
            hSkelText[int(i)-1] = plt.text(xy[0,int(i)-1], xy[1,int(i)-1], strs.cell[int(i)-1])
            
    
    
    xy = xy.conj().T
    #% axis equal
    plt.hold(on)
    for i in np.arange(1., (length(indices))+1):
        if gt:
            hSkel[int(i)-1] = line(np.array(np.hstack((xy[int(I[int(i)-1])-1,0], xy[int(J[int(i)-1])-1,0]))), np.array(np.hstack((xy[int(I[int(i)-1])-1,1], xy[int(J[int(i)-1])-1,1]))), 'color', 'k', 'Parent', hparent)
            set(hSkel[int(i)-1], 'linewidth', linewidth, 'LineStyle', '--')
        else:
            hSkel[int(i)-1] = line(np.array(np.hstack((xy[int(I[int(i)-1])-1,0], xy[int(J[int(i)-1])-1,0]))), np.array(np.hstack((xy[int(I[int(i)-1])-1,1], xy[int(J[int(i)-1])-1,1]))), 'color', M[int(i)-1,:], 'Parent', hparent)
            set(hSkel[int(i)-1], 'linewidth', linewidth)
            
        
        
    return [hSkel, hSkelText]