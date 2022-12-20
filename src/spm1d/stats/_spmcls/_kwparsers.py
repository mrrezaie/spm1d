
import warnings
import inspect


class InferenceKeywordArgumentParser( object ):

	def __init__(self, stat, dim, method):
		self.stat    = stat
		self.dim     = dim
		self.kwargs  = None
		self.method  = self._parse_method(method)
		# self.dirn    = None
		# self.roi     = None   #1D only

	def __repr__(self):
		s    = 'InferenceKeywordArgumentParser\n'
		s   += '   stat     :  %s\n' %self.stat
		s   += '   dim      :  %d\n' %self.dim
		s   += '   kwargs   :  %s\n' %self.kwargs
		# s   += '   dirn     :  %s\n' %self.dirn
		return s


	@property
	def is_0d(self):
		return self.dim == 0
	@property
	def is_1d(self):
		return self.dim == 1
	@property
	def is_t(self):
		return self.stat == 'T'

	def _parse_method(self, s):
		methods = ['param', 'perm', 'fdr']
		if self.dim==1:
			methods += ['rft']
		if s not in methods:
			raise ValueError( f'"method" must be one of: {methods}' )


	def _warn_two_tailed(self):
		msg  = '\n\n\n'
		msg += 'The keyword argument "two_tailed" will be removed from future versions of spm1d.\n'
		msg += '  - Use "dirn=0" in place "two_tailed=True"\n'
		msg += '  - Use "dirn=+1" or "dirn=-1" in place of "two_tailed=False".\n'
		msg += '  - In spm1d v0.4 and earlier, "two_tailed=False" is equivalent to the new syntax: "dirn=1".\n'
		msg += '  - In spm1d v0.5 the default is "dirn=0".\n'
		msg += '\n\n\n'
		warnings.warn( msg , UserWarning, stacklevel=5)

	def parse_dirn_twotailed( self ):
		d = self.kwargs
		
		if ('two_tailed' in d) and ('dirn' in d):
			self._warn_two_tailed()
			two_tailed = d['two_tailed']
			dirn       = d['dirn']
			if two_tailed and (dirn!=0):
				raise ValueError('"dirn" can only be 0 when two_tailed=True')
			if (not two_tailed) and (dirn==0):
				raise ValueError('"dirn" must be -1 or 1 when two_tailed=False')
		elif 'two_tailed' in d:  # and not dirn
			self._warn_two_tailed()
			dirn  = 0 if d['two_tailed'] else 1
		elif 'dirn' in d: # and not two_tailed
			dirn  = d['dirn']
		else:
			dirn  = 0  # default value (two_tailed=True)
		if dirn is None:
			dirn  = 0
		if 'two_tailed' in d:
			d.pop( 'two_tailed' )
		d['dirn']   = dirn
		self.kwargs = d

	def parse_expected_keys(self):
		expected = []
		if self.stat == 'T':
			expected += ['dirn', 'two_tailed']
		if self.method == 'perm':
			expected += ['perms']
		if self.dim == 1:
			expected += ['roi']
		for key in self.kwargs.keys():
			if key not in expected:
				raise KeyError( f'Unexpected key: "{key}".' )
				
		
	def parse_expected_types(self):
		d = self.kwargs
		
		expected = {
			'dirn'       : int,
			'two_tailed' : [False, True],
			'perms'      : int,
			'roi'        : np.ndarray,
		}

	def parse_expected_values(self):
		d = self.kwargs
		
		expected = {
			'dirn'       : [-1, 0, 1, None],
			'two_tailed' : [False, True],
			'perms'      : int,
			'roi'        : np.ndarray,
		}
		
		for k,v in expected.items():
			if v inspect.isclass( v ):
				if not isinstance(k, v):
					raise ValueError(  f'{k} must be an object of type: {v}')
				
				
			if ('dirn' in d) and (d['dirn'] not in [-1, 0, 1, None]):
				raise ValueError('"dirn" must be one of: [-1, 0, 1, None]')

		if ('two_tailed' in d) and (d['two_tailed'] not in [-1, 0, 1, None]):
			raise ValueError('"two_tailed" must be one of: [-1, 0, 1, None]')

		if ('two_tailed' in d) and (d['two_tailed'] not in [-1, 0, 1, None]):
			raise ValueError('"dirn" must be one of: [-1, 0, 1, None]')
		

	def parse(self, **kwargs):
		self.kwargs  = kwargs
		self.parse_expected_keys()
		self.parse_expected_values()
		if self.stat == 'T':
			self.parse_dirn_twotailed()
		

		# self.kwargs  = kwargs
		# self.parse_dirn()
		# self.parse_dirn()



# class KeywordArgumentParserT1(_KeywordArgumentParser):
# 	pass
#
#
#
#
#
# def get_kwarg_parser(stat, dim):
# 	if stat=='T':
# 		parser = KeywordArgumentParserT0() if (dim==0) else KeywordArgumentParserT1()
# 	else:
# 		parser = _KeywordArgumentParser()
	