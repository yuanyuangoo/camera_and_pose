
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def estimateWPCamera(xy, XY):

    # Local Variables: Raff, taff, xy, Inew, camera, A, rinit, numPoints, M, S, R, U, V, X, scaleFactor, b, r3, i, meanX, XY, t, x
    # Function calls: estimateWPCamera, flipud, diag, inv, det, cross, fliplr, abs, zeros, sum, mean, repmat, svd, norm, size
    #%[Raff, taff, R, scale,t,Rproc,tproc]=
    #%estimateOrthoCamera(xy,XY,scaleFactor)
    xy = xy[:,0:2.]
    #% xy=scaleFactor*xy;
    taff = np.mean(xy, 1.)
    xy = xy-matcompat.repmat(taff, np.array(np.hstack((matcompat.size(xy, 1.), 1.))))
    numPoints = matcompat.size(xy, 1.)
    A = np.zeros((2.*numPoints), (matcompat.size(XY, 2.)+3.))
    b = np.zeros((2.*numPoints), 1.)
    for i in np.arange(1., (numPoints)+1):
        A[int(((i-1.)*2.+1.))-1,:] = np.array(np.hstack((XY[int(i)-1,:], 0., 0., 0.)))
        A[int((i*2.))-1,:] = np.array(np.hstack((0., 0., 0., XY[int(i)-1,:])))
        b[int(((i-1.)*2.+1.))-1] = xy[int(i)-1,0]
        b[int((i*2.))-1] = xy[int(i)-1,1]
        
    #%% Procrustes solution as initial point
    x = xy
    X = XY.conj().T
    meanX = np.mean(X, 2.)
    X = X-matcompat.repmat(meanX, 1., matcompat.size(X, 2.))
    M = np.dot(np.dot(x.conj().T, X.conj().T), linalg.inv(np.dot(X, X.conj().T)))
    #% M = x'/X;
    [U, S, V] = plt.svd(M)
    Inew = np.array(np.vstack((np.hstack((1., 0., 0.)), np.hstack((0., 1., 0.)))))
    R = np.dot(np.dot(U, Inew), V.conj().T)
    r3 = np.cross(R[0,:], R[1,:])
    r3 = matdiv(r3, linalg.norm(r3))
    rinit = np.array(np.hstack((R[0,:], R[1,:]))).conj().T
    Raff = R
    R = np.array(np.vstack((np.hstack((R)), np.hstack((r3)))))
    scaleFactor = np.dot(0.5, S[1,1]+S[0,0])
    t = np.array(np.vstack((np.hstack((taff.conj().T)), np.hstack((1./scaleFactor)))))
    plt.det(R)
    if np.sum(np.abs(np.diag(U))) > 1.:
        S = S[0:2.,0:2.]
    else:
        S = S[0:2.,0:2.]
        S = np.flipud(np.fliplr(S))
        #%     disp('Flip scales');
        
    
    #% S = U*S;
    #%%
    #% %Solve non-linear optimization using fmincon
    #% options = optimset('Algorithm','interior-point');
    #% r=fmincon(@(x) objective(x,A,b),rinit,[],[],[],[],[],[], @(x) orthoConstraints(x),options);
    #% % 
    #% Raff=[ r(1) r(2) r(3);...
    #%        r(4) r(5) r(6);];
    #% % 
    #% % Find full rotation matrix
    #% r1=Raff(1,:);
    #% r2=Raff(2,:);
    #% r3=cross(Raff(1,:),Raff(2,:));
    #% r3=r3/norm(r3);
    #% R=[r1;r2;r3];
    #% 
    #% %Estimate average scale
    #% %scale = mean(r2*XY'./xyunscaled(:,2)');
    #% %t=[taff';scale];
    #% 
    camera.Raff = Raff
    camera.taff = taff.conj().T
    camera.R = R
    camera.t = t
    camera.S = S
    camera.M = np.dot(S, Raff)
    return [camera]
def objective(x, A, b):

    # Local Variables: A, x, b, f
    # Function calls: objective
    f = np.dot((np.dot(A, x)-b).conj().T, np.dot(A, x)-b)
    return [f]