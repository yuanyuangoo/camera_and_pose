import numpy as np
import scipy
import matcompat
import posec
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#% [X, R, t] = function recon3DPose(xy,im,varargin)
#%
#% Inputs:   xy - [2 x 14] matrix of 2D joint locations
#%           im - Input image
#%           
#%
#%
#% Outputs:  X  - [3 x 14] matrix of 3D joint locations.
#%           R  - [3 x 3]  Relative Camera Rotation.
#%           t  - [3 x 1]  Relative Camera translation.
#%
#% Wrapper for reconstruction of the 3D Pose of a human figure given the 
#% locations of the 2D anatomical landmarks.
#% Copyright (C) 2012  Varun Ramakrishna.
#% 
#% This program is free software: you can redistribute it and/or modify
#% it under the terms of the GNU General Public License as published by
#% the Free Software Foundation, either version 3 of the License, or
#% (at your option) any later version.
#% 
#% This program is distributed in the hope that it will be useful,
#% but WITHOUT ANY WARRANTY; without even the implied warranty of
#% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#% GNU General Public License for more details.
#% 
#% You should have received a copy of the GNU General Public License
#% along with this program.  If not, see <http://www.gnu.org/licenses/>.
def recon3DPose(im, xy, varargin):

    # Local Variables: Xnew1, annoids, basis, camera, pose, numPoints, mu, skel, xy, im, t, BOMP, varargin, X, R
    # Function calls: load, figure, visualizeGaussianModel, drawCam, clf, pose, setK, length, ones, isempty, alignToCamera, cameraAndPose, recon3DPose, size
    #% [X, R, t] = recon3DPose(xy,im,varargin)
    #% Parse parameters.
    pose=posec()
    [pose.skel, pose.BOMP, pose.mu, pose.lambda1,pose.lamda2, pose.K, pose.numIter,pose.numIters2, pose.tol1, pose.tol2, pose.ks,pose.optType, pose.viz, pose.annoids,pose.numPoints]=pose.process_options('viz',1,'skel','','BOMP','','mu','','lambda2',0.01,'lambda1',0.01,'K',setK(matcompapose.size(im, 2.), matcompapose.size(im, 1.), 2.),'numIter',20.,'numIters2',30.,'tol1',500.,'tol2',1.,'ks',15.,'optType',1.,'viz',0.,'annoids',np.arange(1., 16.0),'numPoints')

    pose.im = im
    pose.xy = np.array(np.vstack((np.hstack((xy)), np.hstack((np.ones(1., matcompat.size(xy, 2.)))))))
    #% Load default basis and skeleton
    if isempty((pose.BOMP)) or isempty((pose.mu)) or isempty((pose.skel)):
        basis = np.load('mocapReducedModel.mat')
        pose.BOMP = basis.B
        pose.mu = basis.mu
        pose.skel = basis.skel
        pose.numPoints = length((pose.skel.tree))
        pose.annoids = np.array(np.hstack((np.arange(1., (length((pose.skel.tree)))+1))))
    
    
    #% Reconstruct camera and pose.
    [camera, pose] = cameraAndPose(pose)
    #% Assign outputs
    X = pose.XnewR
    R = camera.R
    t = camera.t
    #% Show aligned output
    if pose.viz:
        np.load(frontCam)
        Xnew1 = alignToCamera((pose.XnewR), (camera.R), (camera.t), R, t)
        plt.figure(9.)
        plt.clf
        visualizeGaussianModel(Xnew1, (pose.skel))
        drawCam(R, t)
    
    
    return [X, R, t]