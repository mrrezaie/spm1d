
'''
SPM module

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)

This module contains class definitions for raw SPMs (raw test statistic continua)
and inference SPMs (thresholded test statistic).
'''

# Copyright (C) 2022  Todd Pataky


# import sys
# import numpy as np
# from scipy import stats
# import rft1d
# from .. plot import plot_spm, plot_spm_design
# from .. plot import plot_spmi, plot_spmi_p_values, plot_spmi_threshold_label
# from . _clusters import Cluster




# def df2str(df):
# 	return str(df) if not df%1 else '%.3f'%df
# def dflist2str(dfs):
# 	return '(%s, %s)' %(df2str(dfs[0]), df2str(dfs[1]))
# def p2string(p):
# 	return '<0.001' if p<0.0005 else '%.03f'%p
# def plist2string(pList):
# 	s      = ''
# 	if len(pList)>0:
# 		for p in pList:
# 			s += p2string(p)
# 			s += ', '
# 		s  = s[:-2]
# 	return s
# def _set_docstr(childfn, parentfn, args2remove=None):
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







class _SPMParent(object):
	'''Parent class for all parametric SPM classes.'''
	isanova       = False
	isinference   = False
	isregress     = False
	isparametric  = True
	dim           = 0
	testname      = None
	
	
	def _set_testname(self, name):
		self.testname = str( name )



class _SPMF(object):
	'''Additional attrubutes and methods specific to SPM{F} objects.'''
	effect        = 'Main A'
	effect_short  = 'A'
	isanova       = True

	def _repr_summ(self):  # abstract method to be implemented by all subclasses
		pass

	def set_effect_label(self, label=""):
		self.effect        = str(label)
		self.effect_short  = self.effect.split(' ')[1]




















# #set docstrings:
# _set_docstr(_SPM.plot, plot_spm, args2remove=['spm'])
# _set_docstr(_SPMinference.plot, plot_spmi, args2remove=['spmi'])
# _set_docstr(_SPMinference.plot_p_values, plot_spmi_p_values, args2remove=['spmi'])
# _set_docstr(_SPMinference.plot_threshold_label, plot_spmi_threshold_label, args2remove=['spmi'])




