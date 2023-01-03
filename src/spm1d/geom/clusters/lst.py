
'''
ClusterList module
'''

# Copyright (C) 2023  Todd Pataky


from . wrapped import WrappedCluster


class ClusterList(list):

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