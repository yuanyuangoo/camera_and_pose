
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def leastsquareseqcon3(A, b, C, d, alpha):

    # Local Variables: numNorm, xsol, num, alpha, ind, realLambdaSol, Dc, Da, lambdaSol, A, C, U, V, X, a, c, b, e, d, f, i, p, r, den, x, lambda
    # Function calls: max, numden, leastsquareseqcon3, isnan, diag, sum, gsvd, solve, sym, find, length, abs, vpa, double, coeffs, diff, imag, coeff, roots, size
    #% Solve generalized SVD to get gammas and alphas. Please refer Gander et al, for description of the algorithm.
    [U, V, X, Da, Dc] = gsvd(A, C)
    c = np.dot(U.conj().T, b)
    e = np.dot(V.conj().T, d)
    p = matcompat.size(C, 1.)
    r = np.sum((np.diag(Dc) > 0.))
    lambda = sym('lambda', 'real')
    i = 1.
    #% tic;
    f = matdiv(np.dot(Da[int(i)-1,int(i)-1]**2., (np.dot(Dc[int(i)-1,int(i)-1], c[int(i)-1])-np.dot(Da[int(i)-1,int(i)-1], e[int(i)-1]))**2.), (Da[int(i)-1,int(i)-1]**2.+np.dot(lambda, Dc[int(i)-1,int(i)-1]**2.))**2.)
    for i in np.arange(2., (r)+1):
        a = matdiv(np.dot(Da[int(i)-1,int(i)-1]**2., (np.dot(Dc[int(i)-1,int(i)-1], c[int(i)-1])-np.dot(Da[int(i)-1,int(i)-1], e[int(i)-1]))**2.), (Da[int(i)-1,int(i)-1]**2.+np.dot(lambda, Dc[int(i)-1,int(i)-1]**2.))**2.)
        f = f+a
        
    #% toc;
    if length(e) > r:
        for i in np.arange(r+1., (p)+1):
            f = f+e[int(i)-1]**2.
            
    
    
    f = f-alpha**2.
    [num, den] = numden(f)
    c = coeffs(num)
    numNorm = vpa(matdiv(num, c[int(0)-1]))
    lambdaSol = np.roots(coeff)
    if np.diff(f) != 0.:
        lambdaSol = plt.solve(numNorm)
        lambdaSol = np.double(lambdaSol)
        realLambdaSol = lambdaSol[int((not np.abs(np.imag(lambdaSol)) > 0.))-1]
        ind = nonzero((realLambdaSol == matcompat.max(realLambdaSol)))
        ind = ind[0]
        #% xsol = (A'*A+realLambdaSol(ind)*(C'*C))\(A'*b+realLambdaSol(ind)*C'*d);
        xsol = xsol.conj().T
        x = xsol
        if np.sum(np.isnan(xsol)):
            xsol[int(np.isnan[int(xsol)-1])-1] = 0.
        
        
        #% disp(xsol');
    else:
        
        
    
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
    return [x, xsol]