
import numpy as np
from . permuters import get_permuter
from . probcalc0d import ProbCalc0DSingleStat, ProbCalc0DMultiStat



class PermResults0D(object):
	def __init__(self, zc, p, permuter, nperm):
		self.method   = 'perm'
		self.zc       = zc
		self.p        = p
		self.permuter = permuter
		self.nperm    = nperm
		
	def __repr__(self):
		s  = 'PermResults0D\n'
		s += '   zc    = %.5f\n' %self.zc
		s += '   p     = %.5f\n' %self.p
		s += '   nperm = %d\n'   %self.nperm
		return s



def inference0d(stat, z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
	permuter = get_permuter(testname, 0)( *args )
	permuter.build_pdf(  nperm  )
	probcalc = ProbCalc0DSingleStat(stat, permuter, alpha, dirn)
	zc       = probcalc.get_z_critical()
	p        = probcalc.get_p_value(z)
	return PermResults0D(zc, p, permuter, nperm)
	

def inference0d_multi(stat, z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
	permuter = get_permuter(testname, 0)( *args )
	permuter.build_pdf(  nperm  )
	probcalc = ProbCalc0DMultiStat(stat, permuter, alpha, dirn)
	zc       = probcalc.get_z_critical()
	p        = probcalc.get_p_value(z)
	results  = [PermResults0D(zzc, pp, permuter, nperm)   for zzc,pp in zip(zc,p)]
	return results
	
