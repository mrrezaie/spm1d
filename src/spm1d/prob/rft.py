
'''
Common parametric (0D) probabilities
'''

from copy import deepcopy
import numpy as np
from .. geom import assemble_clusters



class RFTResults(object):
	def __init__(self, alpha, dirn, zc, clusters, p_set, p_max):
		self.method   = 'rft'
		self.alpha    = alpha
		self.dirn     = dirn
		self.zc       = zc
		self.clusters = clusters
		self.p_set    = p_set
		self.p_max    = p_max
		self.extras   = {}


def _clusterlevel_inference(calc, z, zc, fwhm, dirn=1, circular=False):
	clusters = assemble_clusters(z, zc, dirn=dirn, circular=circular)
	sc       = 2 if (dirn==0) else 1
	for i,c in enumerate( clusters ):
		k,u  = c.extent / fwhm, abs(c.height)
		p    = sc * calc.p.cluster(k, u)
		clusters[i] = c.as_inference_cluster(k, p)
	return clusters


def _pmax(calc, z, dirn):
	if dirn==0:
		z = np.abs(z).max()
		s = 2
	elif dirn==1:
		z = z.max()
		s = 1
	else:
		z = z.min()
		s = 1
	return s * calc.sf(z)


def _setlevel_inference(calc, zc, clusters):
	c = len(clusters)
	if c==0:
		p = np.nan
	else:
		k = min([c.metric for c in clusters])  # min extent_resels
		p = calc.p.set(c, k, zc)
	return p
	


	
def rft(STAT, z, df, fwhm, resels, alpha=0.05, cluster_size=0, circular=False, withBonf=True, dirn=1, equal_var=False):
	
	'''
	COMMENT:  fwhm is needed ONLY for cluster extent (in FWHM units)
	'''
	import rft1d
	_stats = ["T", "F", "T2", "X2"]
	if STAT not in _stats:
		raise ValueError( f'Unknown statistic: {STAT}. Must be one of: {_stats}' )
	

	if STAT=='T':
		a    = 0.5 * alpha if (dirn==0) else alpha
	else:
		if dirn!=1:
			raise ValueError('"dirn=0" and "dirn=-1" can only be used when STAT=="T"')
		a    = alpha
	calc     = rft1d.prob.RFTCalculatorResels(STAT=STAT, df=df, resels=resels, withBonf=withBonf, nNodes=z.size)
	zc       = calc.isf(a)
	clusters = _clusterlevel_inference(calc, z, zc, fwhm, dirn=dirn, circular=circular)
	p_set    = _setlevel_inference(calc, zc, clusters)
	p_max    = _pmax(calc, z, dirn)
	results  = RFTResults( alpha, dirn, zc, clusters, p_set, p_max )
	return results
	







