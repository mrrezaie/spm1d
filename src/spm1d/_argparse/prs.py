
from . exp import ExpectedArgument, ExpectedKeywordArgument
from . inv import InvalidArgumentCombination, InvalidArgumentPair



class ArgumentParser(object):
	def __init__(self):
		self.ea      = []  # ExpectedArgument
		self.ek      = {}  # ExpectedKeywordArgument
		self.ic      = []  # InvalidArgumentCombination
		
	@property
	def expected_keys(self):
		return list( self.ek.keys() )
	
	
	def _parse_badvalues(self, *args, **kwargs):
		for e,value in zip(self.ea,args):
			e.parse_badvalues( value )
		for key,value in kwargs.items():
			self.ek[key].parse_badvalues( value )

	def _parse_combinations(self, **kwargs):
		for c in self.ic:
			c.parse( **kwargs )

	def _parse_deprecated(self, **kwargs):
		for k in kwargs.keys():
			self.ek[k].warn()

	def _parse_keys(self, **kwargs):
		ekeys = self.expected_keys
		for key,value in kwargs.items():
			if key not in ekeys:
				raise ValueError( f'Unspported keyword argument "{key}".\nSupported keyword arguments: {ekeys}')

	def _parse_ranges(self, *args, **kwargs):
		for e,value in zip(self.ea,args):
			e.parse_range( value )
		for key,value in kwargs.items():
			self.ek[key].parse_range( value )

	def _parse_types(self, *args, **kwargs):
		for e,value in zip(self.ea,args):
			e.parse_type( value )
		for key,value in kwargs.items():
			self.ek[key].parse_type( value )

	def _parse_values(self, *args, **kwargs):
		for e,value in zip(self.ea,args):
			e.parse_value( value )
		for key,value in kwargs.items():
			self.ek[key].parse_value( value )
		

	
	
	def add_arg(self, name, values=None, type=None, range=None, badvalues=None):
		e = ExpectedArgument(name)
		e.set(values, type, range, badvalues)
		self.ea.append( e )
		return e

	def add_kwarg(self, name, values=None, type=None, range=None, default=None, badvalues=None):
		e = ExpectedKeywordArgument(name, default=default)
		e.set(values, type, range, badvalues)
		self.ek[name] = e
		return e
		



	def parse(self, *args, **kwargs):
		# self._parse_args( *args )
		self._parse_keys( **kwargs )
		self._parse_types( *args, **kwargs )
		self._parse_values( *args, **kwargs )
		self._parse_badvalues( *args, **kwargs )
		self._parse_ranges( *args, **kwargs )
		self._parse_deprecated( **kwargs )
		
		# kwargs = self._set_defaults()
		
		
		self._parse_combinations( **kwargs )
		
		
	def set_invalid_combinations(self, names, value_pairs):
		n0,n1  = names
		for x0,x1 in value_pairs:
			c  = InvalidArgumentCombination( n0, n1, x0, x1 )
			self.ic.append( c )
		
	def set_invalid_pairs(self, names):
		n0,n1  = names
		c      = InvalidArgumentPair( n0, n1 )
		self.ic.append( c )
		

