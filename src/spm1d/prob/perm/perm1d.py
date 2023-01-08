
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



class ProbabilityCalculator1D(object):
	def __init__(self, permuter, z, alpha=0.05, dirn=0):
		self.permuter  = permuter
		self.alpha     = alpha
		self.dirn      = dirn
		self.z         = z
		self.zc        = None  # set when "get_z_critical" is called

	@property
	def h0rejected(self):
		if self.dirn==0:
			h   = np.abs(self.z).max() > self.zc
		elif self.dirn==1:
			h   = self.z.max() > self.zc
		else:
			h   = self.z.min() < -self.zc
		return h
	
	@property
	def minp(self):
		return 1 / self.permuter.nperm_actual

	def get_z_critical(self):
		a,dirn,Z = self.alpha, self.dirn, self.permuter.Z
		perc     = 100*(1-0.5*a) if (dirn==0) else 100*(1-a)
		self.zc  = np.percentile(Z, perc, interpolation='midpoint', axis=0)
		return self.zc
	

	def clamp_p_cluster(self, p):
		'''
		Clamp upcrossing p-values to the range (minp, alpha)
		where minp is 1/nperm
		'''
		return max(min(p, self.alpha), self.minp)

	def clamp_p_max(self, p):
		'''
		Clamp field max p-values to the ranges:
			(minp, alpha) if h0 rejected
			(alpha, 1) if h0 not rejected
		where minp is 1/nperm
		'''
		if self.h0rejected:
			p = max(min(p, self.alpha), self.minp)
		else:
			p = max(p, self.alpha)
		return p
	
	def cluster_inference(self, clusters):
		for i,c in enumerate(clusters):
			x = self.permuter.metric.get_single_cluster_metric( c )
			p = (self.permuter.Z2 > x).mean()
			p = self.clamp_p_cluster(p)
			clusters[i] = c.as_inference_cluster( x, p )
		return clusters


	def get_p_max(self):
		'''
		Probability value associated with field maximum
		'''
		z,dirn = self.z, self.dirn
		Z      = self.permuter.Z
		if dirn==0:
			zmx   = max(-z.min(), z.max())
			p     = 2 * ( Z > zmx ).mean()
		elif dirn==1:
			p     = ( Z > z.max() ).mean()
		elif dirn==-1:
			p    = ( Z > -z.min() ).mean()
		p = self.clamp_p_max(p)
		return p


	def setlevel_inference(self, clusters):
		'''
		Set-level inference
		'''
		z,dirn = self.z, self.dirn
		Z2,C2  = self.permuter.Z2, self.permuter.C2
		c      = len( clusters )
		k      = min([self.permuter.metric.get_single_cluster_metric( c )  for c in clusters])
		p      = ((C2>=c) & (Z2>k)).mean()
		p      = self.clamp_p_cluster(p)
		return p


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


# def inference1dff(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000, circular=False, cluster_metric='MaxClusterExtent'):
# 	pass

