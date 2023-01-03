
import numpy as np
eps = np.finfo(float).eps


def estimate_fwhm(r):
	'''
	(Copied from rft1d.geom)
	
	Estimate field smoothness (FWHM) from a set of random fields or a set of residuals.
	
	:Parameters:

		*r* --- a set of random fields or a set of residuals
		
	:Returns:

		*fwhm* --- the estimated FWHM
	
	:Example:
	
		>>> fwhm = 12.5
		>>> y = rft1d.random.randn1d(8, 101, fwhm)
		>>> w = rft1d.geom.estimate_fwhm(y)  # should be close to the specified fwhm value
		
	.. note:: The estimated sample fwhm will differ from the specified population fwhm (just like sample means differ from the population mean). This function implements an unbiased estimate of the fwhm, so the average of many fwhm estimates is expected to converge to the specified value.
	
	'''
	ssq    = (r**2).sum(axis=0)
	# ### gradient estimation (Method 1:  SPM5, SPM8)
	# dx     = np.diff(R, axis=1)
	# v      = (dx**2).sum(axis=0)
	# v      = np.append(v, 0)   #this is likely a mistake but is entered to match the code in SPM5 and SPM8;  including this yields a more conservative estimate of smoothness
	### gradient estimation (Method 2)
	dy,dx  = np.gradient(r)
	v      = (dx**2).sum(axis=0)
	# normalize:
	v     /= (ssq + eps)
	# ### gradient estimation (Method 3)
	# dx     = np.diff(R, axis=1)
	# v      = (dx**2).sum(axis=0)
	# v     /= (ssq[:-1] + eps)
	# ignore zero-variance nodes:
	i      = np.isnan(v)
	v      = v[np.logical_not(i)]
	# global FWHM estimate:
	rpn    = np.sqrt(v / (4*log(2)))  # resels per node
	fwhm   = 1 / rpn.mean()
	return fwhm