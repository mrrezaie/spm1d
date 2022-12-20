#
# # from . _kwparsers import get_kwarg_parser
#
#
# class InferenceManagerParam(object):
# 	def __init__(self, spm, alpha):
# 		self._a     = float(alpha)
# 		self.alpha  = float(alpha)
# 		self.dirn   = None
# 		self.spm    = spm
#
#
# 	def __repr__(self):
# 		s    = 'InferenceManagerParam\n'
# 		s   += '   dirn     :  %s\n' %self.dirn
# 		return s
#
# 	@property
# 	def is_t(self):
# 		return self.spm.STAT=='T'
#
# 	# def _parse_kwargs_dirn_twotailed( self, **kwargs ):
# 	# 	keys       = kwargs.keys()
# 	# 	if ('two_tailed' in keys) and ('dirn' in keys):
# 	# 		self._warn_two_tailed()
# 	# 		two_tailed = kwargs['two_tailed']
# 	# 		dirn       = kwargs['dirn']
# 	# 		if two_tailed and (dirn!=0):
# 	# 			raise ValueError('"dirn" can only be 0 when two_tailed=True')
# 	# 		if (not two_tailed) and (dirn==0):
# 	# 			raise ValueError('"dirn" must be -1 or 1 when two_tailed=False')
# 	# 	elif 'two_tailed' in keys:  # and not dirn
# 	# 		self._warn_two_tailed()
# 	# 		dirn  = 0 if kwargs['two_tailed'] else 1
# 	# 	elif 'dirn' in keys:
# 	# 		dirn  = kwargs['dirn']
# 	# 	else:
# 	# 		dirn  = 0  # default value (two_tailed=True)
# 	# 	if dirn not in [-1, 0, 1]:
# 	# 		raise ValueError('"dirn" must be -1, 0 or 1')
# 	# 	return dirn
#
# 	# def _warn_two_tailed(self):
# 	# 	msg  = '\n\n\n'
# 	# 	msg += 'The keyword argument "two_tailed" will be removed from future versions of spm1d.\n'
# 	# 	msg += '  - Use "dirn=0" in place "two_tailed=True"\n'
# 	# 	msg += '  - Use "dirn=+1" or "dirn=-1" in place of "two_tailed=False".\n'
# 	# 	msg += '  - In spm1d v0.4 and earlier, "two_tailed=False" is equivalent to the new syntax: "dirn=1".\n'
# 	# 	msg += '  - In spm1d v0.5 the default is "dirn=0".\n'
# 	# 	msg += '\n\n\n'
# 	# 	warnings.warn( msg , DeprecationWarning, stacklevel=5)
# 	# 	warnings.warn( msg , UserWarning, stacklevel=5)
#
#
# 	def get_inference_object(self):
# 		pass
# 		# from . _spm0di import SPM0Di
# 		# spmi           = deepcopy( self )
# 		# spmi.__class__ = SPM0Di
# 		#
# 		# spmi._set_inference_params('param', alpha, zc, p, dirn)
# 		# return spmi
#
# 	def parse_kwargs(self, **kwargs):
# 		parser    = InferenceKeywordArgumentParser(self.STAT, self.dim)
# 		parser.parse( **kwargs )
#
# 		# parser   = get_kwarg_parser(self.STAT, self.dim)
# 		# parser.parse( **kwargs )
# 		# dirn     = parser.dirn
# 		#
# 		# # parse dirn and two_tailed
# 		# if self.is_t:
# 		# 	self.dirn = self._parse_kwargs_dirn_twotailed( **kwargs)
# 		# else:
# 		# 	# check for required keyword arguments
# 		# 	pass
# 		#
# 		# # parse two_tailed:
# 		# if 'two_tailed' in keys:
# 		#
# 		#
# 		# # parse direction:
# 		# if self.is_t:
# 		# 	if
# 		# else:
# 		# 	if 'dirn' in keys:
# 		# 		raise ValueError( 'The "two_tailed" option can only be used with T stats.' )
# 		#
# 		# _dirn_default = 0 if (self.is_t) else None
# 		# dirn          = kwargs.get('dirn', _dirn_default)
# 		#
# 		#
# 		#
# 		# keys   = kwargs.keys()
# 		# if 'two_tailed' in keys:
# 		# 	if self.STAT!='T':
# 		# 		raise ValueError( 'The "two_tailed" option can only be used with T stats.' )
#
#
# 	def run(self):
# 		pass
# 		# if self.STAT == 'T':
# 		# 	dirn       = self._kwargs2dirn( **kwargs )
# 		# 	two_tailed = dirn==0
# 		# 	a          = 0.5*alpha if two_tailed else alpha
# 		# 	z          = abs(self.z) if two_tailed else (dirn * self.z)
# 		# 	zc         = stats.t.isf( a, self.df[1] )
# 		# 	p          = stats.t.sf(  z, self.df[1] )
# 		# 	p          = min(1, 2*p) if two_tailed else p
# 		# else:
# 		# 	zc,p   = 0,0
#
