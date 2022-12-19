
'''
Utility functions (general processing)
'''


from math import sqrt,log
import numpy as np
from scipy.ndimage import gaussian_filter1d




def interp(y, Q=101):
	'''
	Simple linear interpolation to *n* values.
	
	:Parameters:
	
	- *y* --- a 1D array or list of J separate 1D arrays
	- *Q* --- number of nodes in the interpolated continuum
	
	:Returns:
	
	- Q-component 1D array or a (J x Q) array
	
	:Example:
	
	>>> y0 = np.random.rand(51)
	>>> y1 = np.random.rand(87)
	>>> y2 = np.random.rand(68)
	>>> Y  = [y0, y1, y2]
	
	>>> Y  = spm1d.util.interp(Y, Q=101)
	'''
	y          = np.asarray(y)
	if (y.ndim==2) or (not np.isscalar(y[0])):
		return np.asarray( [interp(yy, Q)   for yy in y] )
	else:
		x0     = range(y.size)
		x1     = np.linspace(0, y.size, Q)
		return np.interp(x1, x0, y, left=None, right=None)
		
		
		



def smooth(Y, fwhm=5.0):
	'''
	Smooth a set of 1D continua.
	This method uses **scipy.ndimage.filters.gaussian_filter1d** but uses the *fwhm*
	instead of the standard deviation.
	
	:Parameters:
	
	- *Y* --- a (J x Q) numpy array
	- *fwhm* ---  Full-width at half-maximum of a Gaussian kernel used for smoothing.
	
	:Returns:
	
	- (J x Q) numpy array
	
	:Example:
	
	>>> Y0  = np.random.rand(5, 101)
	>>> Y   = spm1d.util.smooth(Y0, fwhm=10.0)
	
	.. note:: A Gaussian kernel's *fwhm* is related to its standard deviation (*sd*) as follows:
	
	>>> fwhm = sd * sqrt(8*log(2))
	'''
	sd    = fwhm / sqrt(8*log(2))
	return gaussian_filter1d(Y, sd, mode='wrap')
