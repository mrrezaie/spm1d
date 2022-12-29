
'''
Common parametric (0D) probabilities
'''

import numpy as np
import rft1d


class ParamResults(object):
	def __init__(self, zc, p):
		self.method   = 'param'
		self.zc = zc
		self.p  = p

def isf_sf_t(z, df, alpha=0.05, dirn=0):
	# critical value:
	a      = 0.5 * alpha if (dirn==0) else alpha
	zc     = rft1d.t.isf0d( a, df )
	zc     = -zc if (dirn==-1) else zc
	# p-value:
	zz     = np.abs(z) if (dirn==0) else dirn*z
	p      = rft1d.t.sf0d( zz, df )
	p      = min(1, 2*p) if (dirn==0) else p
	return zc,p



# def isf_sf_F(z, df, alpha=0.05):
# 	return zc,p


def param(stat, z, df, alpha=0.05, **kwargs):
	if stat=='T':
		df   = df[1] if isinstance(df, (list,tuple,np.ndarray)) else df
		zc,p = isf_sf_t(z, df, alpha, **kwargs)
	elif stat=='F':
		zc   = rft1d.f.isf0d( alpha, df )
		p    = rft1d.f.sf0d( z, df )
	elif stat=='T2':
		pass
	elif stat=='X2':
		pass
	else:
		raise ValueError( f'Unknown statistic: {stat}. Must be one of: ["T", "F", "T2", "X2"]' )
	results = ParamResults( zc, p )
	return results



	### non-sphericity:
	Q        = None
	if not equal_var:
		J           = JA + JB
		q0,q1       = np.eye(JA), np.eye(JB)
		Q0,Q1       = np.matrix(np.zeros((J,J))), np.matrix(np.zeros((J,J)))
		Q0[:JA,:JA] = q0
		Q1[JA:,JA:] = q1
		Q           = [Q0, Q1]