


from . _spm0d import SPM0D
from . _spm1d import SPM1D




#
# def _set_docstr(childfn, parentfn, args2remove=None):
# 	import sys
# 	docstr      =  parentfn.__doc__
# 	if args2remove!=None:
# 		docstrlist0 = docstr.split('\n\t')
# 		docstrlist1 = []
# 		for s in docstrlist0:
# 			if not np.any([s.startswith('- *%s*'%argname) for argname in args2remove]):
# 				docstrlist1.append(s)
# 		docstrlist1 = [s + '\n\t'  for s in docstrlist1]
# 		docstr  = ''.join(docstrlist1)
# 	if sys.version_info.major==2:
# 		childfn.__func__.__doc__ = docstr
# 	elif sys.version_info.major==3:
# 		childfn.__doc__ = docstr



# # set docstrings
# from . _base import _set_docstr
# _set_docstr(_SPM1D.plot, plot_spm, args2remove=['spm'])
# _set_docstr(_SPM1Dinference.plot, plot_spmi, args2remove=['spmi'])
# _set_docstr(_SPM1Dinference.plot_p_values, plot_spmi_p_values, args2remove=['spmi'])
# _set_docstr(_SPM1Dinference.plot_threshold_label, plot_spmi_threshold_label, args2remove=['spmi'])