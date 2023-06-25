

import numpy as np


class Factor(object):
	def __init__(self, A, name='A'):
		self.A            = np.asarray(A, dtype=int)        #integer vector of factor levels
		self.J            = None     # number of observations
		self.u            = None     # unique levels
		self.n            = None     # number of levels
		self.name         = name
		# self.isnested     = False    # nested flag
		# self.nested       = None
		# self.balanced     = True
		self._parse()

	def __repr__(self):
		s  = f'Factor "{self.name}"\n'
		s += f'    J       = {self.J}\n'
		s += f'    n       = {self.n}\n'
		s += f'    u       = {self.u}\n'
		return s
	
	def _parse(self):
		self.J            = self.A.size
		self.u            = np.unique(self.A)
		self.n            = self.u.size

	@property
	def nlevels(self):
		return self.n

	def get_design_main(self):
		X = []
		for u in self.u:
			x = np.zeros(self.J)
			x[self.A==u] =  1
			X.append(x)
		return np.array( X ).T
