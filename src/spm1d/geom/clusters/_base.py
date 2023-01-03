

'''
Clusters base class module
'''

# Copyright (C) 2023  Todd Pataky


from abc import ABCMeta, abstractmethod, abstractproperty
from copy import deepcopy
from ... util import tuple2str, p2string


class _Cluster(metaclass=ABCMeta):
	
	_InferenceClass = None
	iswrapped       = False
	Q               = None
	x               = None
	z               = None
	u               = None
	sign            = None
	
	def __repr__(self):
		s       = f'{self.__class__.__name__}\n'
		s      += '   centroid            :  %s\n'   %tuple2str(self.centroid, '%.3f')
		s      += '   endpoints           :  %s\n'   %tuple2str(self.endpoints, '%.3f')
		s      += '   endpoints_preinterp :  %s\n'   %tuple2str(self.endpoints_preinterp, '%d')
		s      += '   extent              :  %.3f\n' %self.extent
		s      += '   height              :  %.3f\n' %self.height
		s      += '   integral            :  %.3f\n' %self.integral
		s      += '   iswrapped           :  %s\n'   %self.iswrapped
		s      += '   max                 :  %.3f\n' %self.max
		s      += '   min                 :  %.3f\n' %self.min
		s      += '   nnodes              :  %d\n'   %self.nnodes
		s      += '   sign                :  %d\n'   %self.sign
		return s

	@abstractproperty
	def centroid(self):
		pass

	@abstractproperty
	def endpoints(self):
		pass

	@abstractproperty
	def endpoints_preinterp(self):
		pass

	@abstractproperty
	def extent(self):
		pass

	@abstractproperty
	def height(self):
		pass

	@abstractproperty
	def integral(self):
		pass

	@abstractproperty
	def nnodes(self):
		pass

	@abstractproperty
	def min(self):
		pass

	@abstractproperty
	def max(self):
		pass

	@abstractmethod
	def asarray(self):
		pass

	@abstractmethod
	def plot(self):
		pass


	def as_inference_cluster(self, k, p, **kwargs):
		c = deepcopy( self )
		c.__class__ = self._InferenceClass
		c.set_inference_params(k, p, **kwargs)
		return c
		
		# if self.iswrapped:
		# 	self.__class__ = WrappedClusterWithInference
		# else:
		# 	c.__class__ = ClusterWithInference
		# c.set_inference_params( k, p )
		# clusters[i] = c
		
		

class _WithInference(object):
	
	def __repr__(self):
		s   = super().__repr__()
		s  += 'Inference:\n'
		s  += '   extent_resels       :  %.5f\n' %self.extent_resels
		s  += '   p                   :  %s\n'   %p2string(self.p)
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
		
		



