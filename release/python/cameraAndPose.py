
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def cameraAndPose(pose):

    # Local Variables: XMissing, xyMissing, xyt, Mmu, res, pose, repErr, camera, numIters, rigidInds, selectIdx, Msub
    # Function calls: getLimbLengths, eye, estimateWPCamera, scaleInput, kron, length, ones, cameraAndPose, reshapeBasis, inf
    #% Inputs:   pose - Structure containing 2D annotations and algorithm
    #%                  parameters. See recon3DPose.m
    #%
    #% Outputs:  camera - Structure with estimated camera parameters.
    #%           pose   - Structure with estimated 3D pose.
    #%
    #% Reconstructs the 3D Pose of a human figure given the locations of the 2D
    #% anatomical landmarks.
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
    #% Compute lengths and skeleton connectivity matrix.
    pose = getLimbLengths(pose)
    #%% Initialization
    #% Scale input so that K = eye(3).
    pose = scaleInput(pose)
    #% Reshape Basis
    pose = reshapeBasis(pose)
    #% Estimate initial cameras
    rigidInds = np.array(np.hstack((1., 2., 5., 8.)))
    #% Use rigid landmarks to estimate initial cam.
    xyMissing = pose.xyh[:,int(rigidInds)-1]
    XMissing = pose.Xmu[int(rigidInds)-1,0:3.]
    [camera] = estimateWPCamera(xyMissing.conj().T, XMissing)
    #%Assemble projection matrix
    Msub = np.kron(np.eye(length((pose.annoids)), length((pose.annoids))), (camera.M))
    Mmu = np.dot(Msub, pose.mureVis)
    xyt = pose.xyvisible.flatten(1)-np.kron(np.ones(length((pose.annoids)), 1.), (camera.taff))
    res = xyt-Mmu
    #%% Projected Matching Pursuit
    selectIdx = np.array([])
    pose.numIters = 0.
    pose.repErr = np.inf
    #% while (pose.repErr> pose.tol1 && length(selectIdx) < pose.ks)
    #%     %Pick best basis vector
    #%     A = Msub*pose.BreVis;
    #%     A(:,selectIdx) = NaN;
    #%     lambdas = res'*A;
    #%     [~,imax] = max(abs(lambdas));
    #%     % Add basis to collection
    #%     selectIdx = [selectIdx imax];
    #%     pose.selectIdx = selectIdx;
    #%     % Minimize reconstruction error and compute residual
    #%     [camera, pose, res] = minimizeReconError(pose, camera);
    #%     Msub = kron(eye(length(pose.annoids),length(pose.annoids)),camera.M);
    #% end
    return [camera, pose]
def scaleInput(pose):

    # Local Variables: xyunscaled, xyh, pose, xyvisible
    # Function calls: scaleInput
    #% Helper function to scale input so that K = eye(3)
    pose.xyunscaled = pose.xy
    pose.xyh = matdiv(pose.xy, pose.K)
    pose.xyvisible = pose.xyh[0:2.,int((pose.annoids))-1]
    return [pose]
def reshapeBasis(pose):

    # Local Variables: Bnew, mur, i, pose, Bfull, Breshaped
    # Function calls: reshape, cat, reshapeBasis, size
    #% Helper function to reshape basis
    return [pose]
def getLimbLengths(pose):

    # Local Variables: pose, connect
    # Function calls: getLimbLengths, skelConnectionMatrix
    #% Helper function to compute limblengths.
    return [pose]