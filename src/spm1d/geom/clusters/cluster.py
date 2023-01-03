

'''
Clusters module

This module contains class definitions for raw SPMs (raw test statistic continua)
and inference SPMs (thresholded test statistic).
'''

# Copyright (C) 2023  Todd Pataky


import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# import rft1d
from ... util import tuple2str
from . _base import _Cluster



class Cluster( _Cluster ):
	
	iswrapped = False
	
	def __init__(self, x, z, u, sign=1):
		self.Q              = None       # domain size (defined during right-boundary interpolation)
		self.x              = list( x )  # post-interpolated
		self.z              = list( z )  # post-interpolated
		self.u              = u          # threshold
		self.sign           = sign
		self._endpoints     = x[0], x[-1]   # pre-interpolation
		# # pre-interpolated:
		# self._x             = list( x )  # pre-interpolated
		# self._z             = list( z )  # pre-interpolated
		# self._endpoints     = x[0], x[-1]   # pre-interpolation
		
	def __lt__(self, other):
		return self.x[0] < other.x[0]
	
	# def __repr__(self):
	# 	s       = f'{self.__class__.__name__}\n'
	# 	s      += '   centroid            :  %s\n'   %tuple2str(self.centroid, '%.3f')
	# 	s      += '   endpoints           :  %s\n'   %tuple2str(self.endpoints, '%.3f')
	# 	s      += '   endpoints_preinterp :  %s\n'   %tuple2str(self.endpoints_preinterp, '%d')
	# 	s      += '   extent              :  %.3f\n' %self.extent
	# 	s      += '   height              :  %.3f\n' %self.height
	# 	s      += '   integral            :  %.3f\n' %self.integral
	# 	s      += '   isLboundary         :  %s\n'   %self.isLboundary
	# 	s      += '   isRboundary         :  %s\n'   %self.isRboundary
	# 	s      += '   iswrapped           :  %s\n'   %self.iswrapped
	# 	s      += '   max                 :  %.3f\n' %self.max
	# 	s      += '   min                 :  %.3f\n' %self.min
	# 	s      += '   nnodes              :  %d\n'   %self.nnodes
	# 	s      += '   sign                :  %d\n'   %self.sign
	# 	return s


	def _interp_left(self, zf):
		i,u        = self.x[0], self.u
		if i==0:
			self.x = [0] + self.x
			self.z = [u] + self.z
		elif (i>0) and not np.isnan( zf[i-1] ):  # first cluster point not domain edge && previous point not outside ROI
			z0,z1  = zf[i-1], zf[i]
			dx     = (z1-u) / (z1-z0)
			self.x = [i-dx] + self.x
			self.z = [u] + self.z
	
	def _interp_right(self, zf):
		i,u,Q      = self.x[-1], self.u, zf.size
		if i==(Q-1):
			self.x += [Q-1]
			self.z += [u]
		elif (i<(Q-1)) and not np.isnan( zf[i+1] ):  #last cluster point not domain edge && next point not outside ROI
			z0,z1   = zf[i], zf[i+1]
			dx      = (z0-u) / (z0-z1)
			self.x += [i+dx]
			self.z += [u]
		self.Q      = Q


	@property
	def centroid(self):
		x,z = self.asarray().T
		xc  = (x*z).sum() / z.sum()
		zc  = z.mean()
		return xc, zc
	
	@property
	def endpoints(self):
		return self.x[0], self.x[-1]

	@property
	def endpoints_preinterp(self):
		return self._endpoints
	
	@property
	def extent(self):
		x0,x1  = self.endpoints
		w      = x1 - x0
		if w == 0:
			w  = 1
			'''
			LEGACY COMMENTS:
			- when not interpolated, it is possible to have a single-node cluster
			- In this case the extent should be one (and not zero)
			- This case can be removed in future versions IFF all clusters are required to be interpolated
			'''
		return w

	@property
	def height(self):
		return min(self.z) if (self.sign==1) else max(self.z)
	
	@property
	def integral(self):
		x,z = np.asarray(self.x), np.asarray(self.z)
		if x.size==1:
			m = z - self.u
		else:
			m = np.trapz(  z - self.u  )
		return self.sign * m

	@property
	def isLboundary(self):
		return self._endpoints[0] == 0

	@property
	def isRboundary(self):
		return self._endpoints[1] == (self.Q-1)

	@property
	def min(self):
		return min(self.z) if (self.sign==1) else max(self.z)

	@property
	def max(self):
		return max(self.z) if (self.sign==1) else min(self.z)

	@property
	def nnodes(self):
		return len(self.x) - 2  # interpolation adds two points (one to each end)
	
	def asarray(self):
		return np.array( [ self.x, self.z ] ).T
	
	def interp(self, zfull):
		# interpolate to threshold u using similar triangles method
		self._interp_left(zfull)
		self._interp_right(zfull)
		
	def plot(self, ax, **kwargs):
		patch  = Polygon( self.asarray() )
		ax.add_patch( patch, **kwargs )








