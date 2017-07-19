import numpy as np
import scipy
import matcompat
import estimateWPCamera as eW
import skelConnectionMatrix as sCM
import reArrangeBasis as rAB
import leastsquareseqcon3 as ls3
import projectIntoAffineCam as pIA
import plot2Dskeleton as p2s
import drawCam as dC
import cameraclass as cc
import pointsVisualize as pV
import cv2 as cv
def cameraAndPose(pose,camera):

    # Local Variables: XMissing, xyMissing, xyt, Mmu, res, pose, repErr, camera, numIters, rigidInds, selectIdx, Msub
    # Function calls: getLimbLengths, eye, estimateWPCamera, scaleInput, kron, length, ones, cameraAndPose, reshapeBasis, inf
    #Inputs:   pose - Structure containing 2D annotations and algorithm
    #                 parameters. See recon3DPose.m
    #%
    #Outputs:  camera - Structure with estimated camera parameters.
    #          pose   - Structure with estimated 3D pose.
    #%
    #Reconstructs the 3D Pose of a human figure given the locations of the 2D
    #anatomical landmarks.
    #Copyright (C) 2012  Varun Ramakrishna.
    #%
    #This program is free software: you can redistribute it and/or modify
    #it under the terms of the GNU General Public License as published by
    #the Free Software Foundation, either version 3 of the License, or
    #(at your option) any later version.
    #%
    #This program is distributed in the hope that it will be useful,
    #but WITHOUT ANY WARRANTY; without even the implied warranty of
    #MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #GNU General Public License for more details.
    #%
    #You should have received a copy of the GNU General Public License
    #along with this program.  If not, see <http://www.gnu.org/licenses/>.
    #Compute lengths and skeleton connectivity matrix.

    pose=getLimbLengths(pose)
    #%Initialization
    #Scale input so that K = eye(3).
    pose=scaleInput(pose)
    #Reshape Basis
    pose=reshapeBasis(pose)
    #Estimate initial cameras
    rigidInds = np.array((0,1,4,7))
    #Use rigid landmarks to estimate initial cam.
    xyMissing = pose.xyh[:,rigidInds]
    XMissing = pose.Xmu[rigidInds,0:3]
    camera=eW.estimateWPCamera(camera,xyMissing.conj().T, XMissing)
    #%Assemble projection matrix
    Msub = np.kron(np.eye(len(pose.annoids), len(pose.annoids)), camera.M)
    Mmu = np.dot(Msub, pose.mureVis)
    xyt = pose.xyvisible.flatten(1)-np.tile(camera.taff,len(pose.annoids))
    res = xyt - Mmu

    #%Projected Matching Pursuit
    selectIdx = np.array([],int)
    pose.numIters = 0
    pose.repErr = np.inf
    while (pose.repErr > pose.tol1 and len(selectIdx) < pose.ks):

        #Pick best basis vector
        A = np.dot(Msub,pose.BreVis)
        if len(selectIdx):
            A[:, selectIdx] = 0
        lambdas=np.dot(res,A)
        imax=abs(lambdas).argmax()
        # Add basis to collectiona
        print('imax=',imax)
        selectIdx = np.append(selectIdx,imax)
        
        pose.selectIdx = selectIdx

        # Minimize reconstruction error and compute residual
        pose, camera,res = minimizeReconError(pose, camera)
        Msub = np.kron(np.eye(len(pose.annoids), len(pose.annoids)), camera.M)
        print(camera.R)
        print(camera.t)

def scaleInput(pose=None):
    # Helper function to scale input so that K = eye(3)
    pose.xyunscaled = pose.xy
    pose.xyh = np.linalg.lstsq(pose.K,pose.xy)[0]
    pose.xyvisible = pose.xyh[:2,:]
    return pose
def reshapeBasis(pose):
    
    # Local Variables: Bnew, mur, i, pose, Bfull, Breshaped
    # Function calls: reshape, cat, reshapeBasis, size
    #Helper function to reshape basis
    Bnew = np.zeros((pose.BOMP.shape[1],pose.numPoints,3))
    for i in np.arange(0,pose.BOMP.shape[1]):
        tmp=pose.BOMP[:,i].reshape(pose.numPoints,3,order='F').copy()
        Bnew[i] = tmp
    Bnew=Bnew.transpose(1,2,0)

    pose.Breshaped = Bnew
    pose.mur = pose.mu.reshape(pose.numPoints, 3,order='F').copy()
    pose.Bfull = pose.BOMP
    pose.Bre, pose.mure = rAB.reArrangeBasis(pose.BOMP, pose.mu, pose.numPoints)


    pose.BreTensor = pose.Bre.reshape(3,pose.numPoints,pose.Bre.shape[1],order='F')
    pose.mureTensor = pose.mure.reshape(3, pose.numPoints,order='F')
    pose.mureVis = pose.mureTensor[:, pose.annoids]
    pose.mureVis = pose.mureVis.flatten(1)
    pose.BreVis = pose.BreTensor[:,pose.annoids,:]
    pose.BreVis = pose.BreVis.reshape(3 * len(pose.annoids), pose.BreVis.shape[2],order='F')
    return pose
def getLimbLengths(pose):
    connect = sCM.skelConnectionMatrix(pose.skel)
    pose.I, pose.J = np.nonzero(connect)
    
    Xmu = pose.mu.reshape(pose.numPoints, 3,order='F').copy()
    Xmu=np.concatenate((Xmu,np.ones((Xmu.shape[0],1))),axis=1)
    import scipy
    pose.lengths = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Xmu))
    pose.Xmu = Xmu
    #pose.lengths = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Xmu))
    return pose

def minimizeReconError(pose=None, camera=None):
    selectIdx = pose.selectIdx

    optType = pose.optType
    mur = pose.mur
    I = pose.I
    J = pose.J
    lengths = pose.lengths
    Bnew = pose.Breshaped
    xyvisible = pose.xyvisible
    B = pose.Bfull
    annoids = pose.annoids
    epsxy = np.inf
    xyprev = np.zeros((2, pose.numPoints))
    numIters = 0
    
    while (epsxy > pose.tol2 and numIters < pose.numIters2):
        Msub = np.kron(np.eye(len(annoids), len(annoids)), camera.M)
        A = np.dot(Msub,pose.BreVis[:, selectIdx])

        b = xyvisible.flatten(1) - np.tile(camera.taff,len(annoids)) - np.dot(np.kron(np.eye(len(annoids), len(annoids)), camera.M),pose.mureVis)
        #% Equality constrained
        __switch_0__ = optType
        if 0:
            pass
        elif __switch_0__ == 1:
            C = np.array([])
            d = np.array([])
            alphasq = 0
            for i in np.arange(0,len(I)):

                Bi = Bnew[I[i], :, selectIdx].conj().T
                Bj = Bnew[J[i], :, selectIdx].conj().T

                mui = mur[I[i], :]
                muj = mur[J[i], :]

                if i==0:
                    C=Bi-Bj
                    d=-(mui - muj)
                else:
                    C=np.vstack((C,Bi-Bj))
                    d=np.hstack((d,-(mui - muj)))

                alphasq = alphasq + lengths[I[i], J[i]] ** 2

            alpha = np.sqrt(alphasq)
            #alpha = sqrt(pose.lengthsum);
            #C=C.reshape(C.shape[0]/A.shape[1],A.shape[1],order='F')
            d=d.reshape(d.shape[0],1)
            a, asolT = ls3.leastsquareseqcon3(A, b, C, d, alpha)
            asol = asolT.conj().T

        Xnew = np.inner(B[:, selectIdx] , asol).flatten(1) + pose.mu
        XnewR = Xnew.reshape(pose.numPoints, 3,order='F')
        pose.Xnew = Xnew
        pose.XnewR = XnewR
        pose.XnewRt = XnewR.conj().T
        pose.Xnewre = pose.XnewRt.flatten(1)
        pose.XnewreVis = pose.XnewRt[:,annoids]
        pose.XnewreVis = pose.XnewreVis.flatten(1)

        XMissing = XnewR[annoids, :]
        xyMissing = xyvisible
        pose.asol = asol
        pose.selectIdx = selectIdx

        xyrep1 = pIA.projectIntoAffineCam(XnewR, pose.K, camera.R, camera.t, camera.S, pose.skel)
        repErr1=(((xyrep1[0:2, annoids] - pose.xyunscaled[0:2, annoids])** 2).sum(axis=0)).sum(axis=0)
        #Estimate camera
        cameraHyp=cc.camera()
        eW.estimateWPCamera(cameraHyp,xyMissing.conj().T, XMissing)

        #Compute reprojection error
        xyrep2 = pIA.projectIntoAffineCam(XnewR, pose.K, cameraHyp.R, cameraHyp.t, cameraHyp.S, pose.skel)
        repErr2=(((xyrep2[0:2, annoids] - pose.xyunscaled[0:2, annoids])** 2).sum(axis=0)).sum(axis=0)

        camera = cameraHyp
        xyrep = xyrep2
        repErr = repErr2

        epsxy = (((xyrep[0:2, annoids] - xyprev[0:2, annoids])** 2).sum(axis=0)).sum(axis=0)
        xyprev = xyrep
        pose.repErr = repErr

        # Visualize
        if (pose.viz):

            pV.pointsVisualize(XnewR, pose.skel, 'texton', 0)

            dC.drawCam(cameraHyp.R, cameraHyp.t, 'gt', 1)

            

            cv.imshow(pose.im)

            p2s.plot2Dskeleton(xyrep.cT, pose.skel, 1, 'texton', 0)
            p2s.plot2Dskeleton(pose.xyunscaled.cT, pose.skel, 1, 'texton', 0, 'gt', 1)
        

        numIters = numIters + 1
        pose.numIters = pose.numIters + 1

    res = xyvisible.flatten(1) - np.tile(camera.taff,(1,len(annoids))) - np.dot(np.kron(np.eye(len(annoids)),camera.M),pose.XnewreVis)


    res = res - np.dot(np.inner(np.dot(np.kron(np.eye(len(annoids)),camera.M),B[:,selectIdx]),np.dot(np.kron(np.eye(len(annoids)),camera.M),B[:,selectIdx])),res.conj().T).conj().T
    return pose, camera,res