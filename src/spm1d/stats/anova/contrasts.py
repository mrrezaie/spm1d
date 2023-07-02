
import numpy as np
from . factors import Factor



class Contrast(object):
	def __init__(self, C, factors):
		self.C       = C       # contrast matrix
		self.factors = factors # list of Factor objects (used only for factor names)
		

	@property
	def isinteraction(self):
		return self.nfactors > 1
	@property
	def ismain(self):
		return self.nfactors == 1
	@property
	def effect_name(self):
		return 'x'.join( [f.name for f in self.factors] )
	@property
	def effect_name_s(self):
		return 'x'.join( [f.name_s for f in self.factors] )
	@property
	def effect_type(self):
		return 'Main' if self.ismain else 'Interaction'
	@property
	def nfactors(self):
		return len(self.factors)

	@property
	def name(self):
		return f'{self.effect_type} {self.effect_name}'
	
	@property
	def name_s(self):
		return self.effect_name_s
