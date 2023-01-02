
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

# def isf(stat, z, df, resels, alpha=0.05, cluster_size=0, interp=True, circular=False, withBonf=True, dirn=0):
# 	# critical value:
# 	if stat=='T':
# 	else:
# 		a    = alpha
# 	zc     = rft1d.t.isf_resels( a, df, resels, withBonf=withBonf, nNodes=z.size )
# 	print( zc )
# 	return zc
#
# 	# zc     = -zc if (dirn==-1) else zc
# 	# # p-value:
# 	# zz     = np.abs(z) if (dirn==0) else dirn*z
# 	# p      = rft1d.t.sf0d( zz, df )
# 	# p      = min(1, 2*p) if (dirn==0) else p
# 	# return zc,p



# def isf_sf_F(z, df, alpha=0.05):
# 	return zc,p

# alpha=0.05, cluster_size=0, two_tailed=False, interp=True, circular=False, withBonf=True


def rft(stat, z, df, resels, alpha=0.05, cluster_size=0, interp=True, circular=False, withBonf=True, **kwargs):
	
	
	
	
	
	# zc   = isf(stat, z, df, resels, alpha, cluster_size, interp, circular, withBonf, **kwargs)
	if stat=='T':
		# df   = df[1] if isinstance(df, (list,tuple,np.ndarray)) else df
		dirn = kwargs['dirn']
		a    = 0.5 * alpha if (dirn==0) else alpha
		
	elif stat=='F':
		pass
		# zc   = rft1d.f.isf0d( alpha, df )
		# p    = rft1d.f.sf0d( z, df )
	elif stat=='T2':
		pass
	elif stat=='X2':
		pass
	else:
		raise ValueError( f'Unknown statistic: {stat}. Must be one of: ["T", "F", "T2", "X2"]' )
		
	
	calc = rft1d.prob.RFTCalculatorResels(STAT=stat, df=df, resels=resels, withBonf=withBonf, nNodes=z.size)
	zc   = calc.isf(a)
	print( f'zc = {zc}' )


	# results = RFTResults( zc, p )
	# return results



	#
	#
	#
	# ### non-sphericity:
	# Q        = None
	# if not equal_var:
	# 	J           = JA + JB
	# 	q0,q1       = np.eye(JA), np.eye(JB)
	# 	Q0,Q1       = np.matrix(np.zeros((J,J))), np.matrix(np.zeros((J,J)))
	# 	Q0[:JA,:JA] = q0
	# 	Q1[JA:,JA:] = q1
	# 	Q           = [Q0, Q1]