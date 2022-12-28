
import numpy as np
from . permuters import get_permuter


class PermResults(object):
	def __init__(self, zc, p, permuter):
		self.zc       = zc
		self.p        = p
		self.permuter = permuter


def isf_sf_t(z, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
	dim        = 0 if isinstance(z, (int,float)) else np.ndim(z)
	permuter   = get_permuter(testname, dim)( *args )
	permuter.build_pdf(  nperm  )
	zc         = permuter.get_z_critical(alpha, dirn)
	p          = permuter.get_p_value(z, zc, alpha, dirn)
	return zc,p,permuter


def perm(stat, z, alpha=0.05, testname=None, args=None, nperm=10000, **kwargs):
	if stat=='T':
		zc,p,permuter = isf_sf_t(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
	elif stat=='F':
		pass
	elif stat=='T2':
		pass
	elif stat=='X2':
		pass
	else:
		raise ValueError( f'Unknown statistic: {stat}. Must be one of: ["T", "F", "T2", "X2"]' )
	results = PermResults(zc, p, permuter)
	return results