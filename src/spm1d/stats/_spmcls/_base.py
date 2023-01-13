
'''
SPM module

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)

This module contains class definitions for raw SPMs (raw test statistic continua)
and inference SPMs (thresholded test statistic).
'''

# Copyright (C) 2023  Todd Pataky

import warnings
import numpy as np
from ... cfg import SPM1DDeprecationWarning


class _SPM(object):
	dim            = 0
	isinference    = False
	isinlist       = False
	ismultigroup   = False
	testname       = None
	effect_label   = None    # only for ANOVA-like designs
	effect_label_s = None    # only for ANOVA-like designs

	def __eq__(self, other):
		eq = True
		for k,v in self.__dict__.items():
			# print('\n\n\n\n\n')
			# print(k)
			# print('\n\n\n\n\n')
			if not k.startswith('_'):
				v1 = getattr(other, k)
				if v is None:
					eq = v1 is None
				elif isinstance(v, float) and np.isnan(v):
					eq = np.isnan( v1 )
				elif isinstance(v, tuple) and isinstance(v[0], np.ndarray):
					for vv,vv1 in zip(v,v1):
						eq = np.all( np.isclose(vv, vv1, rtol=1e-5, atol=1e-9, equal_nan=True ) )
						if not eq:
							return False
				elif isinstance(v, np.ndarray):
					eq = np.all( np.isclose(v, v1, rtol=1e-5, atol=1e-9, equal_nan=True ) )
				
				elif isinstance(v, (str,int,float,tuple,list,dict)):
					eq = v==v1
				else:
					raise ValueError( f'Unable to hash type: {type(v)}' )
				if not eq:
					return False
		return eq

	@property
	def _class_str(self):
		ss     = 'n' if (self.isinference and not self.isparametric) else ''
		stat   = 't' if (self.STAT=='T') else self.STAT
		s      = f'S{ss}PM{{{stat}}} ({self.dim}D)'
		if self.isinference:
			s += ' inference'
		return s
	@property
	def isanova(self):
		return self.STAT == 'F'
	@property
	def isparametric(self):  # overloaded by inference object (set to False for perm and other nonparametric methods)
		return True
	@property
	def isregress(self):
		return self.testname == 'regress'
	@property
	def ismultigroup(self):
		h = self.testname in ['ttest2', 'anova1']
		if not h:
			h = self.testname.startswith( ('anova2', 'anova3') )
		return h



class _SPMParent(_SPM):
	'''Parent class SPM classes.'''
	
	def _set_anova_attrs(self, ss=None, ms=None):
		self.ss   = tuple( np.asarray(ss, dtype=float) )
		self.ms   = tuple( np.asarray(ms, dtype=float) )
	
	def _set_data(self, *args):
		self._args = args
	
	def _set_testname(self, name):
		self.testname = str( name )

	def set_effect_label(self, s):
		if self.STAT=='F':
			self.effect_label   = s
			self.effect_label_s = s.split(' ')[1]

	# def set_effect_label(self, label=""):
	# 	if self.STAT=='F':
	# 		self.effect        = str(label)
	# 		self.effect_short  = self.effect.split(' ')[1]


# class _SPM0DParent(_SPMParent):
#
# 	def _set_anova_attrs(self, ss=None, ms=None):
# 		self.ss   = tuple( np.asarray(ss, dtype=float) )
# 		self.ms   = tuple( np.asarray(ms, dtype=float) )



class _SPMiParent(_SPM):
	'''Properties for inference SPM classes.'''
	
	isinference = True
	method      = None         # inference method
	alpha       = None         # Type I error rate
	zc          = None         # critical value
	p           = None         # p-value
	dirn        = None         # inference direction (-1, 0 or +1 for T stat, otherwise dirn=1)

	@property
	def _dirn_str(self):
		s     = '0 (two-tailed)' if self.dirn==0 else f'{self.dirn} (one-tailed)'
		return s
		
	# @property
	# def _zcstr(self):
	# 	if isinstance(self.zc, float):
	# 		s  = f'{self.zc:.3f}'
	# 	else:
	# 		z0,z1 = self.zc
	# 		s  = f'{z0:.3f}, {z1:.3f}'
	# 	return s

	@property
	def isparametric(self):
		return self.method in ['param', 'rft', 'fdr', 'bonferroni', 'uncorrected']

	@property
	def eqvar_assumed(self):
		return self.df_adjusted is None

	# @property
	# def nperm_actual(self):
	# 	return self.nperm
	#
	# @property
	# def nperm_possible(self):
	# 	return self.permuter.nPermTotal
	
	@property
	def two_tailed(self):
		return self.dirn == 0

	@property
	def zstar(self):   # legacy support ("zc" was "zstar" in spm1d versions < 0.5)
		msg = 'Use of "zstar" is deprecated. Use "zc" to avoid this warning.'
		warnings.warn( msg , SPM1DDeprecationWarning , stacklevel=2 )
		return self.zc

	
	def _add_extras(self, extras):
		self.extras = extras
		for k,v in extras.items():
			setattr(self, k, v)
	
	
	# def _repr_summ(self):  # abstract method to be implemented by all subclasses
	# 	pass




# class _SPMFParent(object):
# 	'''Additional attrubutes and methods specific to SPM{F} objects.'''
# 	effect        = 'Main A'
# 	effect_short  = 'A'
#
# 	# def _repr_summ(self):  # abstract method to be implemented by all subclasses
# 	# 	pass
#
# 	def set_effect_label(self, label=""):
# 		self.effect        = str(label)
# 		self.effect_short  = self.effect.split(' ')[1]


