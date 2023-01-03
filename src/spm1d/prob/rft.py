
'''
Common parametric (0D) probabilities
'''

from copy import deepcopy
import numpy as np
import rft1d
from .. geom import assemble_clusters



class RFTResults(object):
	def __init__(self, zc, p):
		self.method   = 'rft'
		self.zc = zc
		self.p  = p


def _clusterlevel_inference(calc, z, zc, fwhm, dirn=1, circular=False):
	clusters = assemble_clusters(z, zc, dirn=dirn, circular=circular)
	for i,c in enumerate( clusters ):
		k,u  = c.extent / fwhm, c.height
		p    = calc.p.cluster(k, u)
		clusters[i] = c.as_inference_cluster( k, p )
	return clusters


def _setlevel_inference(calc, zc, clusters):
	c = len(clusters)
	k = clusters.min_extent_resels
	p = calc.p.set(c, k, zc)
	return p
	

	
def rft(STAT, z, df, fwhm, resels, alpha=0.05, cluster_size=0, interp=True, circular=False, withBonf=True, **kwargs):
	
	'''
	COMMENT:  fwhm is needed ONLY for cluster extent (in FWHM units)
	'''
	_stats = ["T", "F", "T2", "X2"]
	if STAT not in _stats:
		raise ValueError( f'Unknown statistic: {STAT}. Must be one of: {_stats}' )
	

	if STAT=='T':
		dirn = kwargs['dirn']
		a    = 0.5 * alpha if (dirn==0) else alpha
	else:
		dirn = 1
		a    = alpha
	calc     = rft1d.prob.RFTCalculatorResels(STAT=STAT, df=df, resels=resels, withBonf=withBonf, nNodes=z.size)
	zc       = calc.isf(a)
	clusters = _clusterlevel_inference(calc, z, zc, fwhm, dirn=dirn, circular=circular)
	p_set    = _setlevel_inference(calc, zc, clusters)
	return zc,clusters,p_set
	


	# spmi       = self._build_spmi(alpha, zstar, clusters, p_set, two_tailed)    #assemble SPMi object


	# results = RFTResults( zc, p )
	# return results



