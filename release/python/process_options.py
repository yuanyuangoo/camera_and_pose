
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#% PROCESS_OPTIONS - Processes options passed to a Matlab function.
#%                   This function provides a simple means of
#%                   parsing attribute-value options.  Each option is
#%                   named by a unique string and is given a default
#%                   value.
#%
#% Usage:  [var1, var2, ..., varn[, unused]] = ...
#%           process_options(args, ...
#%                           str1, def1, str2, def2, ..., strn, defn)
#%
#% Arguments:   
#%            args            - a cell array of input arguments, such
#%                              as that provided by VARARGIN.  Its contents
#%                              should alternate between strings and
#%                              values.
#%            str1, ..., strn - Strings that are associated with a 
#%                              particular variable
#%            def1, ..., defn - Default values returned if no option
#%                              is supplied
#%
#% Returns:
#%            var1, ..., varn - values to be assigned to variables
#%            unused          - an optional cell array of those 
#%                              string-value pairs that were unused;
#%                              if this is not supplied, then a
#%                              warning will be issued for each
#%                              option in args that lacked a match.
#%
#% Examples:
#%
#% Suppose we wish to define a Matlab function 'func' that has
#% required parameters x and y, and optional arguments 'u' and 'v'.
#% With the definition
#%
#%   function y = func(x, y, varargin)
#%
#%     [u, v] = process_options(varargin, 'u', 0, 'v', 1);
#%
#% calling func(0, 1, 'v', 2) will assign 0 to x, 1 to y, 0 to u, and 2
#% to v.  The parameter names are insensitive to case; calling 
#% func(0, 1, 'V', 2) has the same effect.  The function call
#% 
#%   func(0, 1, 'u', 5, 'z', 2);
#%
#% will result in u having the value 5 and v having value 1, but
#% will issue a warning that the 'z' option has not been used.  On
#% the other hand, if func is defined as
#%
#%   function y = func(x, y, varargin)
#%
#%     [u, v, unused_args] = process_options(varargin, 'u', 0, 'v', 1);
#%
#% then the call func(0, 1, 'u', 5, 'z', 2) will yield no warning,
#% and unused_args will have the value {'z', 2}.  This behaviour is
#% useful for functions with options that invoke other functions
#% with options; all options can be passed to the outer function and
#% its unprocessed arguments can be passed to the inner function.
#% Copyright (C) 2002 Mark A. Paskin
#%
#% This program is free software; you can redistribute it and/or modify
#% it under the terms of the GNU General Public License as published by
#% the Free Software Foundation; either version 2 of the License, or
#% (at your option) any later version.
#%
#% This program is distributed in the hope that it will be useful, but
#% WITHOUT ANY WARRANTY; without even the implied warranty of
#% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#% General Public License for more details.
#%
#% You should have received a copy of the GNU General Public License
#% along with this program; if not, write to the Free Software
#% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 1096307
#% USA.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                matcompat.warning(sprintf('Option \'%s\' not used.', args.cell[int(i)-1]))
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