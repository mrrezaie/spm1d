
import numpy as np
from . factors import Factor



class DesignBuilder(object):
	
	def __init__(self, *factors):
		self.factors      = list(factors)
		self.nfactorcols  = []
		# self.XD           = dict(zip(self.labels,  [None]*self.n))

	@property
	def factor_labels(self):
	  return [f.name for f in self.factors]

	@property
	def n(self):
	  return self.nfactors

	@property
	def nfactors(self):
	  return len(self.factors)

	def get_contrasts(self):
		C         = np.zeros( (self.nTerms, self.ncol) )
		for i,col in enumerate(self.COLS):
			C[i,col] = 1
		return Contrasts(C, self.labels)

	def build_design_matrix(self):
		XA        = fA.get_design_mway_main()
		XB        = fB.get_design_mway_main()
		XAB       = fA.get_design_mway_interaction( fB )
		
		
		# X   = np.hstack([self.XD[label]  for label in self.labels])
		# return X
  