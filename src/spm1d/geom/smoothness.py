
from math import log
import numpy as np
eps = np.finfo(float).eps
from . label import bwlabel


def estimate_fwhm(r):
	'''
	(Modified from rft1d.geom)
	'''
	ssq    = (r**2).sum(axis=0)
	### gradient estimation (Method 2)
	dy,dx  = np.gradient(r)
	v      = (dx**2).sum(axis=0)
	# normalize:
	v     /= (ssq + eps)
	# ignore zero-variance nodes:
	i      = np.isnan(v)
	v      = v[np.logical_not(i)]
	# global FWHM estimate:
	rpn    = np.sqrt(v / (4*log(2)))  # resels per node
	fwhm   = 1 / rpn.mean()
	return fwhm
	
	

def _resel_counts(R, fwhm=1, element_based=False):
	'''
	(Modified from rft1d.geom.resel_counts)
	'''
	if R.ndim==2:
		b     = np.any( np.logical_not(np.isnan(R)), axis=0)
	else:
		b     = np.asarray(np.logical_not(R), dtype=bool)
	### Summarize search area geometry:
	nNodes    = b.sum()
	nClusters = bwlabel(b)[1]
	if element_based:
		resels    = nClusters,  float(nNodes)/fwhm
	else:
		resels    = nClusters,  float(nNodes-nClusters)/fwhm
	return resels


def resel_counts(r, fwhm=1, element_based=False, roi=None):
	# define binary search area (False = masked):
	if roi is None:
		resels = _resel_counts(r, fwhm, element_based)
	else:
		b      = np.any( np.isnan(r), axis=0)            # node is True if nan
		b      = np.logical_and(np.logical_not(b), roi)  # node is True if (in roi) and (not nan)
		mask   = np.logical_not(b)                       # True for masked-out regions
		resels = resel_counts(mask, fwhm, element_based)
	return resels