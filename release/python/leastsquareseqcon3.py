import numpy as np
import scipy
import matcompat
import sympy as sy
from sympy import var
import shgsvd as sgh
#from mlabwrap import mlab

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def leastsquareseqcon3(A, b, C, d, alpha):

    # Local Variables: numNorm, xsol, num, alpha, ind, realLambdaSol, Dc, Da, lambdaSol, A, C, U, V, X, a, c, b, e, d, f, i, p, r, den, x, lambda
    # Function calls: max, numden, leastsquareseqcon3, isnan, diag, sum, gsvd, solve, sym, find, length, abs, vpa, double, coeffs, diff, imag, coeff, roots, size
    #% Solve generalized SVD to get gammas and alphas. Please refer Gander et al, for description of the algorithm.
    result=sgh.gsvd(A, C)
    #U,V,X,Da,Dc = mlab.gsvd(A,C,nout=5)
    V=result[5]
    U=result[4]
    Da=result[2]
    
    Dc=result[3]
    X=abs(result[0])
    c = np.dot(U.conj().T, b)
    e = np.dot(V.conj().T, d)
    p = matcompat.size(C, 1.)
    r = np.sum((np.diag(Dc) > 0.))
    lambda1 = var('lambda')
    i = 0
    #% tic;
    
    #f=(Da[i,i]**2)*((Dc[i,i]*c[i]-Da[i,i]*e[i])**2) / ((Da[i,i]**2+lambda1*(Dc[i,i]**2))**2)
    f=(Da[i]**2)*((Dc[i]*c[i]-Da[i]*e[i])**2) / ((Da[i]**2+lambda1*(Dc[i]**2))**2)


    for i in np.arange(1, r):
        a = (Da[i]**2)*((Dc[i]*c[i]-Da[i]*e[i])**2) / ((Da[i]**2+lambda1*(Dc[i]**2))**2)
        f = f+a

    #% toc;
    if len(e) > r:
        for i in np.arange(r, p):
            f = f+(e[i]**2)
            
    
    
    f = f-alpha**2
    from sympy.simplify import combsimp,numer

    num = numer(combsimp(f[0]))
    # coe = num.as_coefficients_dict()
    # coef=np.ones((len(num.as_coefficients_dict())))
    # tmp=0
    # for (k,v) in  coe.items():
    #     coef[tmp]=v
    #     tmp=tmp+1
    # print('n')
    # numNorm = num/c[1]
    #lambdaSol = np.roots(coeff)
    if sy.diff(f) != 0.:
        from sympy import I
        lambdaSol = sy.solve(num)
        realLambdaSol=get_real(lambdaSol)
        #realLambdaSol = lambdaSol[np.logical_not(np.abs(lambdaSol.imag) > 0)]
        ind=realLambdaSol.argmax(0)
        
        up=np.dot(A.conj().T,A) + realLambdaSol[ind]*(np.dot(C.conj().T,C))
        down=np.dot(A.conj().T,b) + realLambdaSol[ind]*np.dot(C.conj().T,d).squeeze()

        #up=up.reshape((up.shape[0],up.shape[0]))
        #down=down.reshape((down.shape[0],1))

        # xsol=np.linalg.solve(up,down)
        if  (up.shape[0]==1):
            xsol=np.array([down[0]/up[0]])
        else:
            xsol,resid,rank,s=np.linalg.lstsq(up,down)

        xsol=xsol.conj().T
        x = xsol
        # if np.sum(np.isnan(xsol)):
        # xsol[np.isnan(xsol)] = 0.

        #% disp(xsol');

    #%% Solve completely symbolically.
    #% syms x lambda;
    #% x = inv(A'*A + lambda*C'*C)*(A'*b + lambda*C'*d);
    #% f = (C*x-d)'*(C*x -d);
    #% lambdaSol = solve(f-alpha^2);
    #% for i = 1:length(lambdaSol)
    #%     xsol(:,i) = feval(matlabFunction(x),double(lambdaSol(i)));
    #% end
    #% x = xsol;
    #% realLambdaSol = lambdaSol(~(abs(imag(lambdaSol))>0));
    #% ind = find(realLambdaSol == max(realLambdaSol));
    #% xsol = x(ind,:);
    return x, xsol
#!/usr/bin/env python

import numpy as np
import scipy.linalg
def get_real(arr):
    for i in np.arange(0,len(arr)):
        if len(arr[i]._args):
            if abs(arr[i]._args[1].args[0])>0.0000001:
                arr[i]=0
            else:
                arr[i]=arr[i]._args[0]
    tmp=np.array(arr)
    tmp=tmp[np.nonzero(tmp)]
    return tmp