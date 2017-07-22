import numpy as np
from scipy.io import loadmat
from setk import setk
from cameraAndPose import cameraAndPose
from alignToCamera import alignToCamera
from visualizeGaussianModel import visualizeGaussianModel
from drawCam import drawCam
def recon3DPose(im,xy,varargin):
    pose={  'lambda2':0.01,
            'lambda1':0.01,
            'K': setK(im.shape),
            'numIter': 20,
            'numIters2':30,
            'tol1': 500,
            'tol2': 1,
            'ks': 15,
            'optType': 1,
            'viz':0,
            'annoids':np.arange(0,15,1),
            'numPoints':15}
    pose['im']=im
    pose['xy']=np.vstack((xy,np.ones(xy.shape[1])))
    if pose.has_key['BOMP'] and pose.has_key['mu'] and pose['skel']:
        basis=loadmat('mocapReducedModel.mat')
        pose['BOMP']=basis['B']
        pose['mu']=basis['mu']
        pose['skel']=basis['skel']
        pose['numPoints']=len(pose['skel']['tree'])
        pose['annoids']=np.arange(0,pose['numPoints'])
    
    camera,pose=cameraAndPose(pose)

    X=pose['XnewR']
    R=camera['R']
    t=camera['t']

    if pose['viz']:
        loadmat(frontCam)
        Xnew1=alignToCamera(pose['XnewR'],camera['R'],camera['t'],R,t)
        visualizeGaussianModel(Xnew1,pose['skel'])
        drawCam(R,t)
    return X,R,t
