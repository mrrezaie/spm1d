

import numpy as np



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
	
	def cluster_inference(self, clusters, Z2=None):
		Z2    = self.permuter.Z2 if (Z2 is None) else Z2
		for i,c in enumerate(clusters):
			x = self.permuter.metric.get_single_cluster_metric( c )
			p = (Z2 > x).mean()
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


class ProbabilityCalculator1DMultiF(ProbabilityCalculator1D):
	
	@property
	def anyh0rejected(self):
		for zz,zzc in zip(self.z, self.zc):
			h = zz.max() > zzc
			if h:
				break
		return h


	def thish0rejected(self, i):
		return self.z[i].max() > self.zc[i]

	def clamp_p_max(self, p, i):
		'''
		Clamp field max p-values to the ranges:
			(minp, alpha) if h0 rejected
			(alpha, 1) if h0 not rejected
		where minp is 1/nperm
		'''
		if self.thish0rejected(i):
			p = max(min(p, self.alpha), self.minp)
		else:
			p = max(p, self.alpha)
		return p

	def get_p_max(self, i):
		'''
		Probability value associated with field maximum
		'''
		z,Z    = self.z[i], self.permuter.Z[:,i]
		p      = ( Z > z.max() ).mean()
		p      = self.clamp_p_max(p, i)
		return p
	
	
	# def cluster_inference(self, Z2, clusters):
	# 	for i,c in enumerate(clusters):
	# 		x = self.permuter.metric.get_single_cluster_metric( c )
	# 		p = (Z2 > x).mean()
	# 		p = self.clamp_p_cluster(p)
	# 		clusters[i] = c.as_inference_cluster( x, p )
	# 	return clusters
		
	# def get_z_critical(self):
	# 	self.zc  =  self.permuter.get_z_critical_list( self.alpha )
	# 	return self.zc
