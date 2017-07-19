import sys 
sys.path.append('../../../hogsvd-python')

import numpy as np
import scipy
import scipy.io as spio

import matcompat
import poseclass as po
import cameraclass as cc

import setK as sk
import cameraAndPose as cp

def recon3DPose(im, xy, varargin):
    pose=po.pose()
    camera=cc.camera()
    pose.im = im

    #process_options(self,skel,BOMP,mu,lambda2,lambda1,K,numIter,numIters2,tol1,tol2,ks,optType,viz,annoids,numPoints):
    pose.process_options('','','',0.01,0.01,sk.setK(im.shape[1], im.shape[0], 2.),20,30,500,1,15,1,0,np.arange(0,15),15)

    #% Load default basis and skeleton
    if not pose.BOMP or not pose.mu or not pose.skel:
        basis = loadmat('./camera_and_pose/release/models/mocapReducedModel.mat')
        #basis = loadmat('../models/mocapReducedModel.mat')

        pose.BOMP = basis['B']
        pose.mu = basis['mu']
        pose.skel = basis['skel']

        pose.numPoints = len(pose.skel['tree'])
        pose.annoids = np.hstack((np.arange(0,pose.numPoints)))
            
    for i in range(len(xy)):
        #xy_=xy[i]
        #xy_=np.transpose(convert(xy_))
        xy_=xy
        
        pose.xy=np.vstack((xy_,np.ones((1,15))))
        #pose.xy=np.concatenate((xy_,np.zeros((1,15))),axis=0)

        #% Reconstruct camera and pose.
        cp.cameraAndPose(pose,camera)
        #% Assign outputs
        X = pose.XnewR
        R = camera.R
        t = camera.t
        #% Show aligned output
        # if pose.viz:
        #     np.load(frontCam)
        #     Xnew1 = alignToCamera((pose.XnewR), (camera.R), (camera.t), R, t)
        #     plt.figure(9.)
        #     plt.clf
        #     visualizeGaussianModel(Xnew1, (pose.skel))
        #     drawCam(R, t)
    
    return X, R, t

def convert(xy=None):
    #index = (6, 7, 5, 2, 3, 4, 1, 8, 9, 9, 15, 14, 13, 10, 11, 12)
    index = (6, 3, 4, 5, 2,0,1,7,9,13,14,15,12,11,10)
    return xy[np.array(index)]


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
