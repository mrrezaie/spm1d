
'''
(OLD) Clusters module

This module contains class definitions for raw SPMs (raw test statistic continua)
and inference SPMs (thresholded test statistic).
'''

# Copyright (C) 2020  Todd Pataky



# from math import floor,ceil
# import numpy as np






# class ClusterNonparam(Cluster):
# 	'''
# 	Suprathreshold cluster (or "upcrossing"):  non-parametric inference
# 	'''
# 	def __init__(self, x, z, u, interp=True):
# 		super(Cluster, self).__init__(x, z, u, interp=interp)
# 		self.name          = 'Cluster (NonParam)'
# 		self.isparam       = False
# 		self.metric        = None
# 		self.iterations    = None
# 		self.nPerm         = None
# 		self.nPermUnique   = None
# 		self.metric_value  = None
# 		self.metric_label  = None
# 		self._assemble()
#
# 	def get_nPermUnique_asstr(self):
# 		n = self.nPermUnique
# 		return n if (n < 1e6) else '%.4g' %n
#
# 	def set_metric(self, metric, iterations, nPermUnique, two_tailed):
# 		self.metric        = metric
# 		self.iterations    = iterations
# 		self.nPerm         = iterations if iterations > 0 else nPermUnique
# 		self.nPermUnique   = nPermUnique
# 		self.metric_label  = metric.get_label_single()
# 		### compute metric value:
# 		x                  = metric.get_single_cluster_metric_xz(self._X, self._Z, self.threshold, two_tailed)
# 		if self.iswrapped:
# 			x             += metric.get_single_cluster_metric_xz(self._other._X, self._other._Z, self.threshold, two_tailed)
# 		self.metric_value  = x
#
# 	def inference(self, pdf, two_tailed):
# 		pdf      = np.asarray(pdf, dtype=float)  # fix for ragged nested sequences
# 		self.P   = (pdf >= self.metric_value).mean()
# 		# self.P             = max( self.P,  1.0/self.nPermUnique )
#




