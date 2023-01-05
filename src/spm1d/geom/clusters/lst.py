
'''
ClusterList module
'''

# Copyright (C) 2023  Todd Pataky


from . wrapped import WrappedCluster


class ClusterList(list):


	@property
	def min_extent(self):
		return min(  [c.extent for c in self]  )

	# @property
	# def min_extent_resels(self):
	# 	if (len(self)>0) and hasattr(self[0], 'extent_resels'):
	# 		e = min(  [c.extent_resels for c in self]  )
	# 	else:
	# 		e = None
	# 	return e

	# @property
	# def min_extent_resels(self):
	# 	if (len(self)>0) and hasattr(self[0], 'extent_resels'):
	# 		e = min(  [c.extent_resels for c in self]  )
	# 	else:
	# 		e = None
	# 	return e

	def asshortstr(self):
		n     = len(self)
		if n==0:
			s = '[]'
		else:
			o = 'objects' if n>1 else 'object'
			s = f'[{n} Cluster {o}]'
		return s
	
	def plot(self, ax, **kwargs):
		for c in self:
			c.plot(ax, **kwargs)

	def sort(self):
		return self.__class__( sorted( self ) )

	def wrap(self):
		c0,c1 = self[0], self[-1]
		if len(self)>1 and c0.isLboundary and c1.isRboundary and (c0.sign==c1.sign):
			self[0] = WrappedCluster( c0, c1 )
			self.pop(-1)