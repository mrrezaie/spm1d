
import numpy as np
from . factors import Factor



class Contrast(object):
	def __init__(self, C, name):
		self.C    = C     # contrast matrix
		self.name = name  # effect label
		
