
import warnings


class SPM1DDeprecationWarning( DeprecationWarning ):
	pass



class SPM1DConfiguration():
	def __init__(self):
		pass
	
	
	def enable_warnings(self, b=True):
		s = 'always' if b else 'ignore'
		warnings.simplefilter(s, SPM1DDeprecationWarning)
	
	
	
	

cfg  = SPM1DConfiguration()