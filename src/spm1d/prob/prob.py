#
#
#
#
# from copy import deepcopy
# import numpy as np
# from scipy import stats
#
#
#
#
# class Param0DCalculator(object):
# 	def __init__(self, STAT, df, alpha=0.05, **kwargs):
# 		self.STAT  = STAT
# 		self.df    = df
# 		self._a    = alpha
# 		self.alpha = alpha
# 		self._parse_kwargs( **kwargs )
#
#
# 	def _parse_kwargs(self, **kwargs):
# 		keys  = kwargs.keys()
# 		if 'two_tailed' in keys():
# 			assert self.STAT == 'T', 'The "two_tailed" option can only be used with T stats.'
#
#
#
# 	def
# 		if self.STAT == 'T'
#
#
# 		a      = 0.5*alpha if two_tailed else alpha
# 		z      = abs(self.z) if two_tailed else self.z
#
#
#
#
# def _kwargs2dirn( **kwargs ):
# 	if ('two_tailed' in keys) and ('dirn' in keys):
# 		if not two_tailed:
# 			if dirn==0:
# 				raise ValueError('"dirn" must be -1 or 1 when two_tailed=False')
# 	elif 'two_tailed' in keys:  # and not dirn
# 		dirn  = 1 if two_tailed else 0
# 	elif 'dirn' in keys:
# 		if dirn not in [-1, 0, 1]:
# 			raise ValueError('"dirn" must be -1, 0 or 1')
# 	else:
# 		dirn  = 0  # default value (two_tailed=False)
# 	return dirn
#
#
# def param_0d(spm, alpha=0.05, **kwargs):
# 	# preliminary keyword argument checks:
# 	keys   = kwargs.keys()
# 	if 'two_tailed' in keys:
# 		if spm.STAT!='T':
# 			raise ValueError( 'The "two_tailed" option can only be used with T stats.' )
# 		# assert self.STAT == 'T', 'The "two_tailed" option can only be used with T stats.'
#
#
# 	spmi   = deepcopy( spm )
#
# 	# calc   = Param0DCalculator(spm.STAT, spm.df, alpha, **kwargs)
#
#
#
#
#
# 	if spm.STAT == 'T':
# 		dirn       = _kwargs2dirn( **kwargs )
# 		two_tailed = dirn==0
# 		a          = 0.5*alpha if two_tailed else alpha
# 		z          = abs(spm.z) if two_tailed else spm.z
# 		zc         = stats.t.isf( a, self.df[1] )
# 		p          = stats.t.sf(  z, self.df[1] )
# 		p          = min(1, 2*p) if two_tailed else p
# 	else:
# 		zc,p   = 0,0
#
#
#
# 	return zc,p
#
#
#