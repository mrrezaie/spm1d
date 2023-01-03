
'''
Common parametric (0D) probabilities
'''

from copy import deepcopy
import numpy as np
import rft1d
from .. geom import assemble_clusters
from .. geom.clusters import Cluster, WrappedCluster, ClusterList
from .. util import p2string


class ParamResults(object):
	def __init__(self, zc, p):
		self.method   = 'param'
		self.zc = zc
		self.p  = p

# def isf(stat, z, df, resels, alpha=0.05, cluster_size=0, interp=True, circular=False, withBonf=True, dirn=0):
# 	# critical value:
# 	if stat=='T':
# 	else:
# 		a    = alpha
# 	zc     = rft1d.t.isf_resels( a, df, resels, withBonf=withBonf, nNodes=z.size )
# 	print( zc )
# 	return zc
#
# 	# zc     = -zc if (dirn==-1) else zc
# 	# # p-value:
# 	# zz     = np.abs(z) if (dirn==0) else dirn*z
# 	# p      = rft1d.t.sf0d( zz, df )
# 	# p      = min(1, 2*p) if (dirn==0) else p
# 	# return zc,p



# def isf_sf_F(z, df, alpha=0.05):
# 	return zc,p

# alpha=0.05, cluster_size=0, two_tailed=False, interp=True, circular=False, withBonf=True

class _WithInference(object):
	
	def __repr__(self):
		s   = super().__repr__()
		s  += 'Inference:\n'
		s  += '   extent_resels       :  %.5f\n' %self.extent_resels
		s  += '   p                   :  %s\n' %p2string(self.p)
		return s
	
	def set_inference_params(self, extent_resels, p, **kwargs):
		self.extent_resels    = extent_resels
		self.p                = p
		self.inference_params = kwargs
	
	# legacy properties:
	@property
	def extentR(self):
		return self.extent_resels
	
	@property
	def P(self):
		return self.p


class ClusterWithInference(_WithInference, Cluster):
	pass

class WrappedClusterWithInference(_WithInference, WrappedCluster):
	pass



def _cluster_inference(cluster, STAT, df, fwhm, resels, two_tailed, withBonf, nNodes):
	extentR = cluster.extent / fwhm
	k,u     = extentR, cluster.height
	if STAT == 'T':
		p = rft1d.t.p_cluster_resels(k, u, df[1], resels, withBonf=withBonf, nNodes=nNodes)
		p = min(1, 2*p) if two_tailed else p
	elif STAT == 'F':
		p = rft1d.f.p_cluster_resels(k, u, df, resels, withBonf=withBonf, nNodes=nNodes)
	elif STAT == 'T2':
		p = rft1d.T2.p_cluster_resels(k, u, df, resels, withBonf=withBonf, nNodes=nNodes)
	elif STAT == 'X2':
		p = rft1d.chi2.p_cluster_resels(k, u, df[1], resels, withBonf=withBonf, nNodes=nNodes)
	
	cluster = deepcopy( cluster )
	if cluster.iswrapped:
		cluster.__class__ = WrappedClusterWithInference
	else:
		cluster.__class__ = ClusterWithInference
	cluster.set_inference_params( extentR, p )
	return cluster
	
	


def rft(stat, z, df, fwhm, resels, alpha=0.05, cluster_size=0, interp=True, circular=False, withBonf=True, **kwargs):
	
	'''
	fwhm is needed ONLY for cluster extent (in FWHM units)
	'''
	
	
	
	
	# zc   = isf(stat, z, df, resels, alpha, cluster_size, interp, circular, withBonf, **kwargs)
	if stat=='T':
		# df   = df[1] if isinstance(df, (list,tuple,np.ndarray)) else df
		dirn = kwargs['dirn']
		a    = 0.5 * alpha if (dirn==0) else alpha
		
	elif stat=='F':
		pass
		# zc   = rft1d.f.isf0d( alpha, df )
		# p    = rft1d.f.sf0d( z, df )
	elif stat=='T2':
		pass
	elif stat=='X2':
		pass
	else:
		raise ValueError( f'Unknown statistic: {stat}. Must be one of: ["T", "F", "T2", "X2"]' )
		
	
	calc     = rft1d.prob.RFTCalculatorResels(STAT=stat, df=df, resels=resels, withBonf=withBonf, nNodes=z.size)
	zc       = calc.isf(a)
	clusters = assemble_clusters(z, zc, dirn=dirn, circular=circular)
	args     = stat, df, fwhm, resels, dirn==0, withBonf, z.size
	clusters = ClusterList(  [_cluster_inference(c, *args) for c in clusters]  )
	
	return zc,clusters
	
	# print( f'zc = {zc}' )
	# print()
	
	# clusters   = self._get_clusters(zstar, check_neg, interp, circular)  #assemble all suprathreshold clusters
	# clusters   = self._cluster_inference(clusters, two_tailed, withBonf)  #conduct cluster-level inference
	# p_set      = self._setlevel_inference(zstar, clusters, two_tailed, withBonf)  #conduct set-level inference
	# spmi       = self._build_spmi(alpha, zstar, clusters, p_set, two_tailed)    #assemble SPMi object


	# results = RFTResults( zc, p )
	# return results



	#
	#
	#
	# ### non-sphericity:
	# Q        = None
	# if not equal_var:
	# 	J           = JA + JB
	# 	q0,q1       = np.eye(JA), np.eye(JB)
	# 	Q0,Q1       = np.matrix(np.zeros((J,J))), np.matrix(np.zeros((J,J)))
	# 	Q0[:JA,:JA] = q0
	# 	Q1[JA:,JA:] = q1
	# 	Q           = [Q0, Q1]