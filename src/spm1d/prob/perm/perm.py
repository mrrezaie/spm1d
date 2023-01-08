
from math import ceil
import numpy as np
from . perm0d import inference0d, inference0d_multi
from . perm1d import inference1d, inference1d_multi




def perm(stat, z, alpha=0.05, testname=None, args=None, nperm=10000, dim=0, **kwargs):
	if alpha < 1/nperm:
		n  = ceil( 1/alpha )
		raise ValueError( f'nperm={nperm} is too small. For alpha={alpha}, nperm must be at least {n}.' )
	
	if dim==0:
		if isinstance(z, (int,float)):
			results = inference0d(stat, z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
		else:
			results = inference0d_multi(stat, z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
	
	elif dim==1:
		
		if z.ndim==1:
			results = inference1d(stat, z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
		else:
			results = inference1d_multi(stat, z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)

	return results
	
	

# def perm(stat, z, alpha=0.05, testname=None, args=None, nperm=10000, dim=0, **kwargs):
# 	if alpha < 1/nperm:
# 		n  = ceil( 1/alpha )
# 		raise ValueError( f'nperm={nperm} is too small. For alpha={alpha}, nperm must be at least {n}.' )
# 	# dim = 0 if isinstance(z, (int,float)) else np.ndim(z)
# 	if dim==0:
# 		results = inference0d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	else:
# 		if testname.startswith( ('anova2','anova3') ):
# 			results = inference1d_mwayanova(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 		else:
# 			results = inference1d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	return results
#
	
# def permff(stat, zz, alpha=0.05, testname=None, args=None, nperm=10000, **kwargs):
# 	if alpha < 1/nperm:
# 		n  = ceil( 1/alpha )
# 		raise ValueError( f'nperm={nperm} is too small. For alpha={alpha}, nperm must be at least {n}.' )
# 	dim = 0 if isinstance(z, (int,float)) else np.ndim(z)
#
#
#
# 	if dim==0:
# 		results = inference0dff(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	else:
# 		results = inference1dff(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	#
# 	# if stat=='T':
# 	# 	if dim==0:
# 	# 		results = inference0d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	# 	else:
# 	# 		results = inference1d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	# elif stat=='F':
# 	# 	if dim==0:
# 	# 		results = inference0d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	# 	else:
# 	# 		results = inference1d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	# elif stat=='T2':
# 	# 	if dim==0:
# 	# 		results = inference0d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	# 	else:
# 	# 		results = inference1d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	# elif stat=='X2':
# 	# 	if dim==0:
# 	# 		results = inference0d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	# 	else:
# 	# 		results = inference1d(z, alpha=alpha, testname=testname, args=args, nperm=nperm, **kwargs)
# 	# else:
# 	# 	raise ValueError( f'Unknown statistic: {stat}. Must be one of: ["T", "F", "T2", "X2"]' )
# 	return results