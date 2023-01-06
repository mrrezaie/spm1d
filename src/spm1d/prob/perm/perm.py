
from math import ceil
import numpy as np
from . perm0d import inference0d
from . perm1d import inference1d




def perm(stat, z, alpha=0.05, testname=None, args=None, nperm=10000, **kwargs):
	if alpha < 1/nperm:
		n  = ceil( 1/alpha )
		raise ValueError( f'nperm={nperm} is too small. For alpha={alpha}, nperm must be at least {n}.' )
	dim = 0 if isinstance(z, (int,float)) else np.ndim(z)
	if stat=='T':
		if dim==0:
			results = inference0d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
		else:
			results = inference1d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
	elif stat=='F':
		pass
	elif stat=='T2':
		pass
	elif stat=='X2':
		pass
	else:
		raise ValueError( f'Unknown statistic: {stat}. Must be one of: ["T", "F", "T2", "X2"]' )
	return results