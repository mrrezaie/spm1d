
import numpy as np
from . permuters import get_permuter



class PermResults0D(object):
	def __init__(self, zc, p, permuter, nperm):
		self.method   = 'perm'
		self.zc       = zc
		self.p        = p
		self.permuter = permuter
		self.nperm    = nperm
		
		
class ProbabilityCalculator0D(object):
	def __init__(self, permuter, alpha=0.05, dirn=0):
		self.permuter  = permuter
		self.alpha     = alpha
		self.dirn      = dirn
		self.zc        = None
		

	def get_p_value(self, z, Z=None, use_both_tails=True):
		Z      = self.permuter.Z if (Z is None) else Z
		dirn   = self.dirn
		# if use_both_tails:   this can only be used for symmetrical distributions
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
				p0     = ( Z >  abs(z) ).mean()
				p1     = ( Z < -abs(z) ).mean()
				p      = p0 + p1
			else:
				if z > 0:
					p  = 2 * ( Z > z ).mean()
				else:
					p  = 2 * ( Z < z ).mean()
		elif dirn==1:
			p          = ( Z > z ).mean()
		elif dirn==-1:
			p          = ( Z < z ).mean()
		return p
		
	
	def get_z_critical(self):
		a,dirn,Z = self.alpha, self.dirn, self.permuter.Z
		# if two-tailed, calculate average of upper and lower thresholds:
		if dirn==0:
			perc0 = 100*(0.5*a)
			perc1 = 100*(1-0.5*a)
			z0,z1 = np.percentile(a, [perc0,perc1], interpolation='midpoint')
			zc    = 0.5 * (-z0 + z1) # must be a symmetrical distribution about zero for two-tailed inference
		else:
			perc  = 100*(1-0.5*a) if (dirn==0) else 100*(1-a)
			zc    = np.percentile(Z, perc, interpolation='midpoint')
		self.zc   = zc
		return zc



def inference0d(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
	permuter = get_permuter(testname, 0)( *args )
	permuter.build_pdf(  nperm  )

	# permuter.set_alpha( alpha )
	# permuter.set_dirn( dirn )
	probcalc = ProbabilityCalculator0D(permuter, alpha, dirn)
	zc       = probcalc.get_z_critical()
	p        = probcalc.get_p_value(z, use_both_tails=True)
	# zc       = permuter.get_z_critical(alpha, dirn)
	# p        = permuter.get_p_value(z, zc, alpha, dirn)
	return PermResults0D(zc, p, permuter, nperm)