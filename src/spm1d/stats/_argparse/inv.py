


class InvalidArgumentCombination(object):
	def __init__(self, name0, name1, value0, value1):
		self.name0   = name0 
		self.name1   = name1
		self.value0  = value0
		self.value1  = value1
		
	def parse(self, **kwargs):
		n0,n1        = self.name0, self.name1
		if (n0 in kwargs) and (n1 in kwargs):
			x0,x1    = kwargs[n0], kwargs[n1]
			v0,v1    = self.value0, self.value1
			if (x0==v0) and (x1==v1):
				raise ValueError( f'"{n1}" cannot be "{x1}" when {n0}={x0}' )
				
				
				

class InvalidArgumentPair(object):
	def __init__(self, name0, name1):
		self.name0   = name0 
		self.name1   = name1
		
	def parse(self, **kwargs):
		n0,n1        = self.name0, self.name1
		if (n0 in kwargs) and (n1 in kwargs):
			raise KeyError( f'"{n0}" and "{n1}" are equivalent keywords; only one can be used.' )