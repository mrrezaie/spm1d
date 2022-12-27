
import numpy as np
from . permuters import get_permuter


def perm_T(z, df, alpha=0.05, dirn=0, testname=None, args=None, nperm=10000):
	dim        = 0 if isinstance(z, (int,float)) else np.ndim(z)
	permuter   = get_permuter(testname, dim)( *args )
	permuter.build_pdf(  nperm  )
	zc         = permuter.get_z_critical(alpha, dirn)
	
	# p = None
	
	p          = permuter.get_p_value(z, zc, alpha, dirn)
	

	# print(permuter)
	# print(permuter.nPermTotal)
	
	# zc,p       = None, None
	
	return zc,p
	
	
	# if not isinstance(zc, float):
	# 	zc     = zc.flatten()
	# p          = permuter.get_p_value(self.z, zc, alpha, dirn)
	#
	# from . _spm0di import SPM0Di
	# spmi             = deepcopy( self )
	# spmi.__class__   = SPM0Di
	# spmi.method      = 'perm'
	# spmi.alpha       = alpha
	# spmi.zc          = zc
	# spmi.p           = p
	# spmi.dirn        = dirn
	# spmi.nperm       = nperm
	# spmi.permuter    = permuter
	# # spmi._set_inference_params('perm', alpha, zc, p, dirn)
	# return spmi