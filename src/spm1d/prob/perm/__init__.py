
import numpy as np
from . permuters import get_permuter
from ... geom import assemble_clusters






class PermResults0D(object):
	def __init__(self, zc, p, permuter, nperm):
		self.method   = 'perm'
		self.zc       = zc
		self.p        = p
		self.permuter = permuter
		self.nperm    = nperm

class PermResults1D(object):
	def __init__(self, zc, clusters, p_max, p_set, permuter, nperm):
		self.method   = 'perm'
		self.zc       = zc
		self.clusters = clusters
		self.p_set    = p_set
		self.p_max    = p_max
		self.permuter = permuter
		self.nperm    = nperm


# def _clusterlevel_inference(permuter, z, zc, fwhm, dirn=1, circular=False):
# 	clusters = assemble_clusters(z, zc, dirn=dirn, circular=circular)
#
# 	# for i,c in enumerate( clusters ):
# 	# 	k,u  = c.extent / fwhm, c.height
# 	# 	p    = calc.p.cluster(k, u)
# 	# 	clusters[i] = c.as_inference_cluster( k, p )
# 	return clusters
#



def isf_sf_t_0d(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
	permuter = get_permuter(testname, 0)( *args )
	permuter.build_pdf(  nperm  )
	zc       = permuter.get_z_critical(alpha, dirn)
	p        = permuter.get_p_value(z, zc, alpha, dirn)
	return PermResults0D(zc, p, permuter, nperm)

def isf_sf_t_1d(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000, circular=False, cluster_metric='MaxClusterExtent'):
	permuter  = get_permuter(testname, 1)( *args )
	permuter.build_pdf(  nperm  )
	zc        = permuter.get_z_critical(alpha, dirn)
	# build secondary PDF:
	permuter.set_metric( cluster_metric )
	permuter.build_secondary_pdf( zc, circular )
	# clusters = _clusterlevel_inference(permuter, z, zc, )
	
	clusters = assemble_clusters(z, zc, dirn=dirn, circular=circular)
	
	for i,c in enumerate( clusters ):
		x    = permuter.metric.get_single_cluster_metric( c )
		p    = permuter.get_cluster_pvalue( x )
		clusters[i] = c.as_inference_cluster( x, p )
	
	
	
	# clusters = self._cluster_inference(clusters, two_tailed)
	# clusters = None
	p_max    = permuter.get_p_max( z, zc, alpha, dirn=dirn )
	p_set    = None
	results  = PermResults1D(zc, clusters, p_max, p_set, permuter, nperm)
	return results

def perm(stat, z, alpha=0.05, testname=None, args=None, nperm=10000, **kwargs):
	dim         = 0 if isinstance(z, (int,float)) else np.ndim(z)
	if stat=='T':
		if dim==0:
			results = isf_sf_t_0d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
		else:
			results = isf_sf_t_1d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
	elif stat=='F':
		pass
	elif stat=='T2':
		pass
	elif stat=='X2':
		pass
	else:
		raise ValueError( f'Unknown statistic: {stat}. Must be one of: ["T", "F", "T2", "X2"]' )
	return results