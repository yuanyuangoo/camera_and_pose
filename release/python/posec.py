import matcompat
import setK
class posec:
    def prt(self):
        print(self)
        print(self.__class__)
    def process_options(args, varargin,nargout=1):
            # Local Variables: unused, nunused, i, args, varargout, n, warn, varargin, found, j, nout
    # Function calls: process_options, nargout, cell, length, warning, sprintf, error, strcmpi, mod
    #% Check the number of input arguments
        n = len(varargin)
        if np.mod(n, 2.):
            matcompat.error('Each option must be a string/value pair.')
    
    
        #% Check the number of supplied output arguments
        if nargout<n/2.:
            matcompat.error('Insufficient number of output arguments given')
        elif nargout == n/2.:
            warn = 1.
            nout = n/2.
            
        else:
            warn = 0.
            nout = n/2.+1.
            
        
        #% Set outputs to be defaults
        varargout = cell(1., nout)
        for i in np.arange(2., (n)+(2.), 2.):
            varargout.cell[int((i/2.))-1] = varargin.cell[int(i)-1]
            
        #% Now process all arguments
        nunused = 0.
        for i in np.arange(1., (length(args))+(2.), 2.):
            found = 0.
            for j in np.arange(1., (n)+(2.), 2.):
                if strcmpi(args.cell[int(i)-1], varargin.cell[int(j)-1]):
                    varargout.cell[int(((j+1.)/2.))-1] = args.cell[int((i+1.))-1]
                    found = 1.
                    break
                
                
                
            if not found:
                if warn:
                    matcompat.warning(sprintf('Option ''%s'' not used.', args.cell[int(i)-1]))
                    args.cell[int(i)-1]
                else:
                    nunused = nunused+1.
                    unused.cell[int((2.*nunused-1.))-1] = args.cell[int(i)-1]
                    unused.cell[int((2.*nunused))-1] = args.cell[int((i+1.))-1]
                    
        #% Assign the unused arguments
        if not warn:
            if nunused:
                varargout.cell[int(nout)-1] = unused
            else:
                varargout.cell[int(nout)-1] = cell(0.)
                
        return [varargout]