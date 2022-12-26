
import warnings



class ExpectedArgument(object):
	def __init__(self, name):
		self.badvalues   = None
		self.name        = name
		self.range       = None
		self.values      = None
		self.type        = None
		
	
	def _msg_badvalue(self, x):
		return f'Invalid value: {x}; argument "{self.name}" must NOT be one of: {self.badvalues}'
	def _msg_range(self, x):
		return f'Invalid value: {x}; argument "{self.name}" must have a value in range: {self.range}'
	def _msg_type(self, x):
		return f'Invalid type: {type(x).__name__}; argument "{self.name}" must be of type: {self.type.__name__}'
	def _msg_value(self, x):
		return f'Invalid value: {x}; argument "{self.name}" value must be one of: {self.values}'
	
	def append_values(self, x):
		self.values += x
	
	def parse_badvalues(self, x):
		if self.badvalues is not None:
			if x in self.badvalues:
				raise ValueError( self._msg_badvalue(x) )
	
	def parse_range(self, x):
		if self.range is not None:
			x0,x1        = self.range
			if (x<x0) or (x>x1):
				raise ValueError( self._msg_range(x) )

	def parse_type(self, x):
		if self.type is not None:
			if not isinstance(x, self.type):
				raise TypeError( self._msg_type(x) )

	def parse_value(self, x):
		if self.values is not None:
			if x not in self.values:
				raise ValueError( self._msg_value(x) )

	def set(self, values=None, type=None, range=None, badvalues=None):
		if values is not None:
			self.values  = values 
		if type is not None:
			self.type    = type
		if range is not None:
			self.range   = range
		if badvalues is not None:
			self.badvalues = badvalues
	#
	# def set_range(self, range):
	# 	self.range   = range
	# def set_type(self, type):
	# 	self.type    = type
	def set_values(self, values):
		self.values  = values
	



class ExpectedKeywordArgument(ExpectedArgument):
	def __init__(self, name, default=None):
		super().__init__(name)
		self.default     = default
		self.warning_msg = None
	
	def _msg_badvalue(self, x):
		return f'Invalid keyword argument value: "{self.name}={x}"; value must NOT be one of: {self.badvalues}'
	def _msg_range(self, x):
		return f'Invalid keyword argument value: "{self.name}={x}"; value must be in range: {self.range}'
	def _msg_type(self, x):
		return f'Invalid type: {type(x).__name__}; keyword argument "{self.name}" must have a value of type: {self.type.__name__}'
	def _msg_value(self, x):
		return f'Invalid keyword argument value: "{self.name}={x}"; value must be one of: {self.values}'

	
		
	def set_deprecated_warning(self, msg):
		self.warning_msg = msg
		
	def warn(self):
		if self.warning_msg is not None:
			# warnings.warn( self.warning_msg , UserWarning, stacklevel=5)
			warnings.warn( self.warning_msg , DeprecationWarning, stacklevel=5)

