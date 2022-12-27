
'''
SPM module

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)

This module contains class definitions for raw SPMs (raw test statistic continua)
and inference SPMs (thresholded test statistic).
'''

# Copyright (C) 2022  Todd Pataky

import warnings



class _SPMParent(object):
	'''Parent class SPM classes.'''
	dim           = 0
	isinference   = False
	isinlist      = False
	
	@property
	def _class_str(self):
		ss   = 'n' if (self.isinference and not self.isparametric) else ''
		stat = 't' if (self.STAT=='T') else self.STAT
		return f'S{ss}PM{{{stat}}} ({self.dim}D)'
	@property
	def isanova(self):
		return self.STAT == 'F'
	@property
	def isregress(self):
		return self.testname == 'regress'
	
	def _set_data(self, *args):
		self._args = args
	
	def _set_testname(self, name):
		self.testname = str( name )


class _SPMiParent(object):
	'''Additional properties for inference SPM classes.'''
	
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
		
	@property
	def _zcstr(self):
		if isinstance(self.zc, float):
			s  = f'{self.zc:.3f}'
		else:
			z0,z1 = self.zc
			s  = f'{z0:.3f}, {z1:.3f}'
		return s

	@property
	def isparametric(self):
		return self.method in ['gauss', 'rft', 'fdr', 'bonferroni', 'uncorrected']

	@property
	def zstar(self):   # legacy support ("zc" was "zstar" in spm1d versions < 0.5)
		msg = 'Use of "zstar" is deprecated. Use "zc" to avoid this warning.'
		warnings.warn( msg , DeprecationWarning , stacklevel=2 )
		return self.zc



class _SPMF(object):
	'''Additional attrubutes and methods specific to SPM{F} objects.'''
	effect        = 'Main A'
	effect_short  = 'A'

	def _repr_summ(self):  # abstract method to be implemented by all subclasses
		pass

	def set_effect_label(self, label=""):
		self.effect        = str(label)
		self.effect_short  = self.effect.split(' ')[1]


