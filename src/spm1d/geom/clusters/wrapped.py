
'''
Wrapped Clusters module
'''

# Copyright (C) 2023  Todd Pataky


import numpy as np
# from ... util import tuple2str
from . _base import _Cluster, _WithInference
# from . inference import WrappedClusterWithInference


class WrappedCluster( _Cluster ):

	iswrapped       = True

	def __init__(self, c0, c1):
		self._InferenceClass = WrappedClusterWithInference
		self.c0             = c0
		self.c1             = c1
		self.Q              = c0.Q             # domain size (defined during right-boundary interpolation)
		self.x              = c0.x + c1.x      # post-interpolated
		self.z              = c0.z + c1.z      # post-interpolated
		self.u              = c0.u             # threshold
		self.sign           = c0.sign
		self._endpoints     = c1._endpoints[0], c0._endpoints[-1]  # pre-interpolation

	@property
	def centroid(self):
		x,z   = self.asarray( modular=False ).T
		xc    = x.mean() % self.Q
		zc    = z.mean()
		return xc, zc

	@property
	def endpoints(self):
		return self.c1.x[0], self.c0.x[-1]

	@property
	def endpoints_preinterp(self):
		return self.c1._endpoints[0], self.c0._endpoints[1]

	
	@property
	def extent(self):
		return self.c0.extent + self.c1.extent
	
	@property
	def height(self):
		return min(self.z) if (self.sign==1) else max(self.z)
	
	@property
	def integral(self):
		return self.c0.integral + self.c1.integral

	@property
	def min(self):
		return min(self.z) if (self.sign==1) else max(self.z)

	@property
	def max(self):
		return max(self.z) if (self.sign==1) else min(self.z)
	
	@property
	def nnodes(self):
		return self.c0.nnodes + self.c1.nnodes

	def asarray(self, modular=False):
		x0,z0 = self.c0.asarray().T
		x1,z1 = self.c1.asarray().T
		x     = np.hstack( [x1, self.Q + x0] )
		z     = np.hstack( [z1, z0] )
		if modular:
			x = x % self.Q
		return np.vstack( [x,z] ).T


	def plot(self, ax, **kwargs):
		self.c0.plot(ax, **kwargs)
		self.c1.plot(ax, **kwargs)



class WrappedClusterWithInference(_WithInference, WrappedCluster):
	pass




