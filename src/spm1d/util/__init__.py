
'''
Utility functions
'''

# Copyright (C) 2022  Todd Pataky


from . prob import p_corrected_bonf, p_critical_bonf
from . proc import interp, smooth
from . str import df2str, dflist2str, p2string, plist2string, plist2stringlist







def get_dataset(*args):
	'''
	.. warning:: Deprecated

		**get_dataset** is deprecated and will be removed from future versions of **spm1d**.  Please access datasets using the "spm1d.data" interface.
	'''
	raise( DeprecationWarning('"get_dataset" is deprecated.  Access datasets using "spm1d.data"') )
