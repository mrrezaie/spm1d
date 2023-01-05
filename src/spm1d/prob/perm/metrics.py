
# Copyright (C) 2023  Todd Pataky

import numpy as np
# from ... geom import bwlabel
from ... geom import assemble_clusters


class _Metric(object):
	pass
	

class MaxClusterExtent(_Metric):
	def get_label(self):
		return self.__class__.__name__

	def get_label_single(self):
		return self.__class__.__name__.strip('Max')

	# def get_all_clusters(self, z, zstar, Z, two_tailed=False):
	# 	clusters      = assemble_clusters()
	# 	return clusters

	def get_all_cluster_metrics(self, z, zc, dirn=1, circular=False):
		clusters = assemble_clusters(z, zc, dirn=dirn, circular=circular)
		return [c.extent for c in clusters]

	def get_max_metric(self, z, zc, dirn=1, circular=False):
		x = self.get_all_cluster_metrics(z, zc, dirn, circular)
		return 0 if (len(x)==0) else max(x)

	def get_single_cluster_metric(self, cluster):
		return cluster.extent

	# def get_single_cluster_metric_xz(self, x, z, zstar, two_tailed=False):
	# 	return x.max() - x.min()
		
		
		

# class _Metric(object):
# 	def get_label(self):
# 		return self.__class__.__name__
# 	def get_label_single(self):
# 		return self.__class__.__name__.strip('Max')
# 	def get_all_clusters(self, z, zstar, Z, two_tailed=False):
# 		clusters      = []
# 		mlabel        = self.get_label_single()
# 		L,n           = bwlabel( z > zstar )
# 		for i in range(n):
# 			m         = self.get_single_cluster_metric(z, zstar, L==i+1)
# 			cluster   = ClusterNP(np.arange(z.size), z, zstar, interp=True, mvalue=m, mlabel=mlabel)
# 			clusters.append( cluster )
# 		return clusters
#
# 	def get_all_cluster_metrics(self, z, thresh=3.0, circular=False):
# 		L,n     = bwlabel(z>thresh)
# 		x       = [0]
# 		if n > 0:
# 			x   = [self.get_single_cluster_metric(z, thresh, L==i+1)   for i in range(n)]
# 			if circular and (n > 1):  #merge clusters for circular fields:
# 				if (L==1)[0] and (L==n)[-1]:
# 					x[0] += x[-1]
# 					x     = x[:-1]
# 		return x
# 	def get_max_metric(self, z, thresh=3.0, circular=False):
# 		return max(  self.get_all_cluster_metrics(z, thresh, circular)  )


# class MaxClusterExtent(_Metric):
# 	def get_single_cluster_metric(self, z, thresh, i):
# 		return i.sum()
# 	def get_single_cluster_metric_xz(self, x, z, zstar, two_tailed=False):
# 		return x.max() - x.min()


class MaxClusterHeight(_Metric):
	def get_single_cluster_metric(self, z, thresh, i):
		return z[i].max()
	def get_single_cluster_metric_xz(self, x, z, zstar, two_tailed=False):
		return z.max()


class MaxClusterIntegral(_Metric):
	def get_single_cluster_metric(self, z, thresh, i):
		if i.sum()==1:
			x = z[i] - thresh
		else:
			x = np.trapz(  z[i]-thresh  )
		return x

	def get_single_cluster_metric_xz(self, x, z, zstar, two_tailed=False):
		if x.size==1:
			m = float(z - zstar)
		else:
			m = np.trapz(  z - zstar  )
		if two_tailed and (m < 0):
			m *= -1
		return m





metric_dict  = {
	'MaxClusterExtent'   : MaxClusterExtent(),
	'MaxClusterHeight'   : MaxClusterHeight(),
	'MaxClusterIntegral' : MaxClusterIntegral()
	}




