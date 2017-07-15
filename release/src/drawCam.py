
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def drawCam(R, t, varargin):

    # Local Variables: gt, scale, P1, maxp, O, Cmid, P, R, Ly, t, cameraPlane, faces, varargin, C2, C1, C4, Lz, Lx, C3
    # Function calls: max, patch, hsv, process_options, line, repmat, drawCam
    [gt] = process_options(varargin, 'gt', 0.)
    scale = 5.
    P = np.dot(scale, np.array(np.vstack((np.hstack((0., 0., 0.)), np.hstack((0.5, 0.5, 0.8)), np.hstack((0.5, -0.5, 0.8)), np.hstack((-0.5, 0.5, 0.8)), np.hstack((-0.5, -0.5, 0.8))))))
    #%P = scale*[0 0 0;0.5 0.5 -0.8; 0.5 -0.5 -0.8; -0.5 0.5 -0.8;-0.5 -0.5 -0.8];
    P1 = np.dot(R.conj().T, P.conj().T-matcompat.repmat(t, np.array(np.hstack((1., 5.)))))
    #%P1=R*P'+repmat(t,[1,5]);
    P1 = P1.conj().T
    maxp = matcompat.max(matcompat.max(P1))
    #%axis(2*[-maxp maxp -maxp maxp -maxp maxp]);
    if not gt:
        line(np.array(np.hstack((P1[0,0], P1[1,0]))), np.array(np.hstack((P1[0,2], P1[1,2]))), np.array(np.hstack((P1[0,1], P1[1,1]))), 'color', 'k')
        line(np.array(np.hstack((P1[0,0], P1[2,0]))), np.array(np.hstack((P1[0,2], P1[2,2]))), np.array(np.hstack((P1[0,1], P1[2,1]))), 'color', 'k')
        line(np.array(np.hstack((P1[0,0], P1[3,0]))), np.array(np.hstack((P1[0,2], P1[3,2]))), np.array(np.hstack((P1[0,1], P1[3,1]))), 'color', 'k')
        line(np.array(np.hstack((P1[0,0], P1[4,0]))), np.array(np.hstack((P1[0,2], P1[4,2]))), np.array(np.hstack((P1[0,1], P1[4,1]))), 'color', 'k')
        line(np.array(np.hstack((P1[1,0], P1[2,0]))), np.array(np.hstack((P1[1,2], P1[2,2]))), np.array(np.hstack((P1[1,1], P1[2,1]))), 'color', 'k')
        line(np.array(np.hstack((P1[2,0], P1[4,0]))), np.array(np.hstack((P1[2,2], P1[4,2]))), np.array(np.hstack((P1[2,1], P1[4,1]))), 'color', 'k')
        line(np.array(np.hstack((P1[4,0], P1[3,0]))), np.array(np.hstack((P1[4,2], P1[3,2]))), np.array(np.hstack((P1[4,1], P1[3,1]))), 'color', 'k')
        line(np.array(np.hstack((P1[3,0], P1[1,0]))), np.array(np.hstack((P1[3,2], P1[1,2]))), np.array(np.hstack((P1[3,1], P1[1,1]))), 'color', 'k')
        cameraPlane = np.array(np.vstack((np.hstack((P1[1,0], P1[1,2], P1[1,1])), np.hstack((P1[3,0], P1[3,2], P1[3,1])), np.hstack((P1[2,0], P1[2,2], P1[2,1])), np.hstack((P1[4,0], P1[4,2], P1[4,1])))))
        faces = np.array(np.hstack((2., 1., 3., 4.)))
        patch('Vertices', cameraPlane, 'Faces', faces, 'FaceVertexCData', plt.hsv(6.), 'FaceColor', 'k', 'FaceAlpha', 0.1)
    else:
        line(np.array(np.hstack((P1[0,0], P1[1,0]))), np.array(np.hstack((P1[0,2], P1[1,2]))), np.array(np.hstack((P1[0,1], P1[1,1]))), 'color', 'k', 'LineStyle', '--')
        line(np.array(np.hstack((P1[0,0], P1[2,0]))), np.array(np.hstack((P1[0,2], P1[2,2]))), np.array(np.hstack((P1[0,1], P1[2,1]))), 'color', 'k', 'LineStyle', '--')
        line(np.array(np.hstack((P1[0,0], P1[3,0]))), np.array(np.hstack((P1[0,2], P1[3,2]))), np.array(np.hstack((P1[0,1], P1[3,1]))), 'color', 'k', 'LineStyle', '--')
        line(np.array(np.hstack((P1[0,0], P1[4,0]))), np.array(np.hstack((P1[0,2], P1[4,2]))), np.array(np.hstack((P1[0,1], P1[4,1]))), 'color', 'k', 'LineStyle', '--')
        line(np.array(np.hstack((P1[1,0], P1[2,0]))), np.array(np.hstack((P1[1,2], P1[2,2]))), np.array(np.hstack((P1[1,1], P1[2,1]))), 'color', 'k', 'LineStyle', '--')
        line(np.array(np.hstack((P1[2,0], P1[4,0]))), np.array(np.hstack((P1[2,2], P1[4,2]))), np.array(np.hstack((P1[2,1], P1[4,1]))), 'color', 'k', 'LineStyle', '--')
        line(np.array(np.hstack((P1[4,0], P1[3,0]))), np.array(np.hstack((P1[4,2], P1[3,2]))), np.array(np.hstack((P1[4,1], P1[3,1]))), 'color', 'k', 'LineStyle', '--')
        line(np.array(np.hstack((P1[3,0], P1[1,0]))), np.array(np.hstack((P1[3,2], P1[1,2]))), np.array(np.hstack((P1[3,1], P1[1,1]))), 'color', 'k', 'LineStyle', '--')
        #%     
        #%     cameraPlane =[P1(2,1) P1(2,3) P1(2,2);  P1(4,1) P1(4,3) P1(4,2); P1(3,1) P1(3,3) P1(3,2);P1(5,1) P1(5,3) P1(5,2)];
        #%     faces =[2 1 3 4];
        #%     patch('Vertices',cameraPlane,'Faces',faces,'FaceVertexCData',hsv(6),'FaceColor','k','FaceAlpha',0.05);
        
    
    C1 = np.array(np.hstack((P1[1,0], P1[1,2], P1[1,1])))
    C2 = np.array(np.hstack((P1[2,0], P1[2,2], P1[2,1])))
    C3 = np.array(np.hstack((P1[3,0], P1[3,2], P1[3,1])))
    C4 = np.array(np.hstack((P1[4,0], P1[4,2], P1[4,1])))
    O = np.array(np.hstack((P1[0,0], P1[0,2], P1[0,1])))
    Cmid = np.dot(0.25, C1+C2+C3+C4)
    #% Lz = [O; O+0.5*(Cmid-O)];
    #% Lx = [O; O+0.5*(C2-C1)];
    #% Ly = [O; O+0.5*(C3-C1)];
    Lz = np.array(np.vstack((np.hstack((O)), np.hstack((O+np.dot(0.5, Cmid-O))))))
    Lx = np.array(np.vstack((np.hstack((O)), np.hstack((O+np.dot(0.5, C1-C3))))))
    Ly = np.array(np.vstack((np.hstack((O)), np.hstack((O+np.dot(0.5, C1-C2))))))
    if not gt:
        line(Lz[:,0], Lz[:,1], Lz[:,2], 'color', 'b', 'linewidth', 2.)
        line(Lx[:,0], Lx[:,1], Lx[:,2], 'color', 'g', 'linewidth', 2.)
        line(Ly[:,0], Ly[:,1], Ly[:,2], 'color', 'r', 'linewidth', 2.)
    else:
        line(Lz[:,0], Lz[:,1], Lz[:,2], 'color', 'b', 'linewidth', 1.)
        line(Lx[:,0], Lx[:,1], Lx[:,2], 'color', 'g', 'linewidth', 1.)
        line(Ly[:,0], Ly[:,1], Ly[:,2], 'color', 'r', 'linewidth', 1.)
        
    
    plt.axis(tight)
    return 