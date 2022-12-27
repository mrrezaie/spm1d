
import numpy as np
import rft1d


def _gauss_T(z, df, alpha=0.05, dirn=0):
	# critical value:
	a      = 0.5 * alpha if (dirn==0) else alpha
	zc     = rft1d.t.isf0d( a, df )
	zc     = -zc if (dirn==-1) else zc
	# p-value:
	zz     = np.abs(z) if (dirn==0) else dirn*z
	p      = rft1d.t.sf0d( zz, df )
	p      = min(1, 2*p) if (dirn==0) else p
	return zc,p


def gauss(stat, z, df, alpha=0.05, **kwargs):
	pass


# def inference(z, df, alpha=0.05, method='gauss', **kwargs):
# 	# parser   = InferenceArgumentParser0D(self.STAT, method)
# 	# parser.parse( alpha, **kwargs )
#
# 	if method == 'gauss':
# 		spmi = self.inference_gauss(alpha, **parser.kwargs)
# 	elif method == 'perm':
# 		spmi = self.inference_perm(alpha, **parser.kwargs)
# 	return spmi




