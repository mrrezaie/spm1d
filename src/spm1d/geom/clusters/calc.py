
'''
ClusterList module
'''

# Copyright (C) 2023  Todd Pataky


import numpy as np
import rft1d
from . cluster import Cluster
from . lst import ClusterList


class _SignedClusterCalculator(object):
	def __init__(self, z, u, sign=1):
		self.L             = None                  # labeled domain
		self.clusters      = []                    # list of cluster objects
		self.sign          = sign                  # inference direction (-1 or +1)
		self.n             = None                  # number of clusters 
		self.x             = np.arange( z.size )   # domain position
		self.z             = z                     # test stat
		self.u             = u                     # (critical) threshold
		if sign==-1:
			self.u         *= -1

	def assemble_clusters(self):
		inds      = []
		for i in range(self.n):
			b     = self.L == (i+1)
			c     = Cluster(self.x[b], self.z[b], self.u, sign=self.sign)
			self.clusters.append( c )

	def interp(self):
		for c in self.clusters:
			c.interp(self.z)
	
	def label(self):
		b      = (self.z >= self.u) if (self.sign==1) else (self.z <= self.u)
		L,n    = rft1d.geom.bwlabel(b)
		self.L = L
		self.n = n



class ClusterCalculator(object):
	def __init__(self, z, u, dirn=0):
		self.dirn     = dirn
		self.z        = z
		self.u        = u
		self.calcpos  = None
		self.calcneg  = None
		self.clusters = None
		self._init_calculators()
	
	def _init_calculators(self):
		if self.dirn == 0:
			self.calcpos = _SignedClusterCalculator(self.z, self.u, sign=1)
			self.calcneg = _SignedClusterCalculator(self.z, self.u, sign=-1)
		elif self.dirn == 1:
			self.calcpos = _SignedClusterCalculator(self.z, self.u, sign=1)
		elif self.dirn == -1:
			self.calcneg = _SignedClusterCalculator(self.z, self.u, sign=-1)

	def assemble_clusters(self):
		c0,c1 = [],[]
		if self.calcpos is not None:
			self.calcpos.assemble_clusters()
			c0 = self.calcpos.clusters
		if self.calcneg is not None:
			self.calcneg.assemble_clusters()
			c1 = self.calcneg.clusters
		self.clusters = ClusterList( c0 + c1 ).sort()

	def label(self):
		if self.calcpos is not None:
			self.calcpos.label()
		if self.calcneg is not None:
			self.calcneg.label()

	def interp(self):
		for c in self.clusters:
			c.interp(self.z)



def assemble_clusters():
	pass
