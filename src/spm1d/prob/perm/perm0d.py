
import numpy as np
from . permuters import get_permuter
from . probcalc0d import ProbabilityCalculator0D


class PermResults0D(object):
	def __init__(self, zc, p, permuter, nperm):
		self.method   = 'perm'
		self.zc       = zc
		self.p        = p
		self.permuter = permuter
		self.nperm    = nperm
		
		



def inference0d(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
	permuter = get_permuter(testname, 0)( *args )
	permuter.build_pdf(  nperm  )
	probcalc = ProbabilityCalculator0D(permuter, alpha, dirn)
	zc       = probcalc.get_z_critical()
	p        = probcalc.get_p_value(z, use_both_tails=True)
	return PermResults0D(zc, p, permuter, nperm)
	

def inference0d_multi_f(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
	pass

# def inference0dff(zz, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
# 	print('in inference0d')
# 	permuter = get_permuter(testname, 0)( *args )
# 	permuter.build_pdf(  nperm  )
# 	probcalc = ProbabilityCalculator0D(permuter, alpha, dirn)
# 	zc       = probcalc.get_z_critical()
# 	p        = probcalc.get_p_value(z, use_both_tails=True)
# 	return PermResults0D(zc, p, permuter, nperm)