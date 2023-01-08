
import numpy as np
from . permuters import get_permuter
from ... geom import assemble_clusters, ClusterList


class PermResults1D(object):
	def __init__(self, zc, clusters, p_max, p_set, permuter, nperm):
		self.method   = 'perm'
		self.zc       = zc
		self.clusters = clusters
		self.p_set    = p_set
		self.p_max    = p_max
		self.permuter = permuter
		self.nperm    = nperm





def inference1d(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000, circular=False, cluster_metric='MaxClusterExtent'):
	permuter  = get_permuter(testname, 1)( *args )
	# build primary PDF:
	permuter.build_pdf(  nperm  )
	probcalc = ProbabilityCalculator1D(permuter, z, alpha, dirn)
	zc       = probcalc.get_z_critical()
	if probcalc.h0rejected:
		# build secondary PDF:
		permuter.set_metric( cluster_metric )
		permuter.build_secondary_pdf( zc, circular )
		# cluster-level inference:
		clusters = assemble_clusters(z, zc, dirn=dirn, circular=circular)
		clusters = probcalc.cluster_inference( clusters )
		# set-level inference:
		p_set    = probcalc.setlevel_inference( clusters )
	else:
		clusters = ClusterList( [] )
		p_set    = None
	# maximum inference:
	p_max    = probcalc.get_p_max()
	results  = PermResults1D(zc, clusters, p_max, p_set, permuter, nperm)
	return results





def inference1d_multi_f(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000, circular=False, cluster_metric='MaxClusterExtent'):
	permuter  = get_permuter(testname, 1)( *args )
	# build primary PDF:
	permuter.build_pdf(  nperm  )
	probcalc = ProbabilityCalculator1DMultiF(permuter, z, alpha, dirn)
	zc       = probcalc.get_z_critical()
	
	
	
	if probcalc.anyh0rejected:
		# build secondary PDF:
		permuter.set_metric( cluster_metric )
		permuter.build_secondary_pdfs( zc, circular )
		
	
	results = []
	for i,(zz,zzc) in enumerate( zip(z,zc) ):
		if zz.max() > zzc:
			# cluster-level inference:
			clusters = assemble_clusters(zz, zzc, dirn=1, circular=circular)
			clusters = probcalc.cluster_inference( clusters, Z2=permuter.Z2[i] )
			# set-level inference:
			p_set    = None
			# p_set    = probcalc.setlevel_inference( clusters )
		else:
			clusters = ClusterList( [] )
			p_set    = None
		p_max    = probcalc.get_p_max( i )
		res      = PermResults1D(zzc, clusters, p_max, p_set, permuter, nperm)
		results.append( res )
			
			
	return results




# def inference1dff(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000, circular=False, cluster_metric='MaxClusterExtent'):
# 	pass

