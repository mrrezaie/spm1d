

import numpy as np


			# if _use_both_tails: # using both tails tends to produce numerically more stable results
			# 	p0     = ( Z >  abs(z) ).mean()
			# 	p1     = ( Z < -abs(z) ).mean()
			# 	p      = p0 + p1
			# else:
			# 	if z > 0:
			# 		p  = 2 * ( Z > z ).mean()
			# 	else:
			# 		p  = 2 * ( Z < z ).mean()


class ProbCalc0DSingleStat(object):
	def __init__(self, stat, permuter, alpha=0.05, dirn=0):
		self.STAT      = stat
		self.permuter  = permuter
		self.alpha     = alpha
		self.dirn      = dirn
		

	def get_p_value(self, z):
		Z,dirn     = self.permuter.Z, self.dirn
		if self.STAT=='T':  # use both tails for more accurate results
			p0     = ( Z >  abs(z) ).mean()
			p1     = ( Z < -abs(z) ).mean()
			p      = p0 + p1
			if dirn in [1,None]:
				p  = (0.5 * p)  if (z>0) else (1 - 0.5*p)
			elif dirn == -1:
				p  = (0.5 * p)  if (z<0) else (1 - 0.5*p)
		else:
			p      = ( Z > z ).mean()
		return p
		
	
	def get_z_critical(self):
		a,dirn,Z = self.alpha, self.dirn, self.permuter.Z
		# if two-tailed, calculate average of upper and lower thresholds:
		if dirn==0:
			perc0 = 100*(0.5*a)
			perc1 = 100*(1-0.5*a)
			z0,z1 = np.percentile(Z, [perc0,perc1], interpolation='midpoint')
			zc    = 0.5 * (-z0 + z1) # must be a symmetrical distribution about zero for two-tailed inference
		else:
			perc  = 100*(1-a)
			zc    = np.percentile(Z, perc, interpolation='midpoint')
		return zc



class ProbCalc0DMultiStat(ProbCalc0DSingleStat):
	
	def get_p_value(self, z):
		p      = ( self.permuter.Z > z ).mean(axis=0)
		return p

	def get_z_critical(self):
		perc     = 100*(1-self.alpha)
		zc       = np.percentile(self.permuter.Z, perc, interpolation='midpoint', axis=0)
		return zc