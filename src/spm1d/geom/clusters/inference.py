

# from . cluster import Cluster
# from . wrapped import WrappedCluster
# from ... util import p2string
#
#
# class _WithInference(object):
#
# 	def __repr__(self):
# 		s   = super().__repr__()
# 		s  += 'Inference:\n'
# 		s  += '   extent_resels       :  %.5f\n' %self.extent_resels
# 		s  += '   p                   :  %s\n'   %p2string(self.p)
# 		return s
#
# 	def set_inference_params(self, extent_resels, p, **kwargs):
# 		self.extent_resels    = extent_resels
# 		self.p                = p
# 		self.inference_params = kwargs
#
# 	# legacy properties:
# 	@property
# 	def extentR(self):
# 		return self.extent_resels
#
# 	@property
# 	def P(self):
# 		return self.p
#
#
# class ClusterWithInference(_WithInference, Cluster):
# 	pass
#
# class WrappedClusterWithInference(_WithInference, WrappedCluster):
# 	pass