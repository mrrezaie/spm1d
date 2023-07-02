
import numpy as np
from . factors import Factor



class Contrast(object):
	def __init__(self, C, name, name_s):
		self.C      = C       # contrast matrix
		self.name   = name    # effect label
		self.name_s = name_s  # effect label (short) (for summary table)
		
