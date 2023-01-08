

import numpy as np




class ProbabilityCalculator0D(object):
	def __init__(self, permuter, alpha=0.05, dirn=0):
		self.permuter  = permuter
		self.alpha     = alpha
		self.dirn      = dirn
		self.zc        = None
		

	def get_p_value(self, z, Z=None, use_both_tails=True):
		Z      = self.permuter.Z if (Z is None) else Z
		dirn   = self.dirn
		# if use_both_tails and issymmetrical:   this can only be used for symmetrical distributions
		# 	p0     = ( Z >  abs(z) ).mean()
		# 	p1     = ( Z < -abs(z) ).mean()
		# 	p2     = p0 + p1
		# 	p      = 0.5 * p2
		# 	if dirn==0:
		# 		p  = p2
		# 	elif dirn==1:
		# 		p  = p if z>0 else (1-p)
		# 	elif dirn==-1:
		# 		p  = p if z<0 else (1-p)
		if dirn==0:
			if use_both_tails: # using both tails tends to produce numerically more stable results
				p0     = ( Z >  abs(z) ).mean(axis=0)
				p1     = ( Z < -abs(z) ).mean(axis=0)
				p      = p0 + p1
			else:
				if z > 0:
					p  = 2 * ( Z > z ).mean(axis=0)
				else:
					p  = 2 * ( Z < z ).mean(axis=0)
		elif dirn==1:
			p          = ( Z > z ).mean(axis=0)
		elif dirn==-1:
			p          = ( Z < z ).mean(axis=0)
		return p
		
	
	def get_z_critical(self):
		a,dirn,Z = self.alpha, self.dirn, self.permuter.Z
		# if two-tailed, calculate average of upper and lower thresholds:
		if dirn==0:
			perc0 = 100*(0.5*a)
			perc1 = 100*(1-0.5*a)
			z0,z1 = np.percentile(Z, [perc0,perc1], interpolation='midpoint', axis=0)
			zc    = 0.5 * (-z0 + z1) # must be a symmetrical distribution about zero for two-tailed inference
		else:
			perc  = 100*(1-0.5*a) if (dirn==0) else 100*(1-a)
			zc    = np.percentile(Z, perc, interpolation='midpoint', axis=0)
		self.zc   = zc
		return zc


