
'''
SPM-0D module

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)

This module contains class definitions for raw 0D SPMs.
'''

# Copyright (C) 2022  Todd Pataky


from copy import deepcopy
import warnings
import numpy as np
from scipy import stats
# from . _base import _SPMParent, _SPMF
from .. import prob
from ... util import dflist2str





class SPM0D(object):

	dim           = 0
	isinference   = False

	def __init__(self, STAT, z, df, beta=None, residuals=None, sigma2=None):
		self.STAT           = STAT             # test statistic ("T" or "F")
		self.z              = float(z)         # test statistic
		self.df             = df               # degrees of freedom
		self.beta           = beta             # fitted parameters
		self.residuals      = residuals        # model residuals
		self.sigma2         = sigma2           # variance
		self.testname       = None             # hypothesis test name (set using set_testname method)
		
		
	
	
		def _set_testname(self, name):
			self.testname = str( name )


	def __repr__(self):
		stat     = 't' if self.STAT=='T' else self.STAT
		s        = ''
		s       += 'SPM{%s} (0D)\n' %stat
		s       += '   SPM.testname :  %s\n'      %self.testname
		if self.isanova:
			s   += '   SPM.effect     :  %s\n'      %self.effect
			s   += '   SPM.SS         : (%s, %s)\n' %self.ss
			s   += '   SPM.df         : (%s, %s)\n' %self.df
			s   += '   SPM.MS         : (%s, %s)\n' %self.ms
			s   += '   SPM.z          :  %.5f\n'    %self.z
		else:
			s   += '   SPM.z        :  %.5f\n'      %self.z
			s   += '   SPM.df       :  %s\n'        %dflist2str(self.df)
		if self.isregress:
			s   += '   SPM.r        :  %.5f\n'      %self.r
		s       += '\n'
		return s


	@property
	def isanova(self):
		return self.STAT == 'F'

	@property
	def isregress(self):
		return self.testname == 'regress'


	def _set_testname(self, name):
		self.testname = str( name )
		
	def _warn_two_tailed(self):
		msg  = '\n\n\n'
		msg += 'The keyword argument "two_tailed" will be removed from future versions of spm1d.\n'
		msg += '  - Use "dirn=0" in place "two_tailed=True"\n'
		msg += '  - Use "dirn=+1" or "dirn=-1" in place of "two_tailed=False".\n'
		msg += '  - In spm1d v0.4 and earlier, "two_tailed=False" is equivalent to the new syntax: "dirn=1".\n'
		msg += '  - In spm1d v0.5 the default is "dirn=0".\n'
		msg += '\n\n\n'
		warnings.warn( msg , DeprecationWarning, stacklevel=5)
		warnings.warn( msg , UserWarning, stacklevel=5)

	# def _inference_param(self, a, z, two_tailed=True):
	# 	if self.STAT == 'T':
	# 		zc     = stats.t.isf( a, self.df[1] )
	# 		p      = stats.t.sf(  z, self.df[1] )
	# 	else:
	# 		zc,p   = 0,0
	# 	p      = min(1, 2*p) if two_tailed else p
	# 	return zc,p

	def _kwargs2dirn( self, **kwargs ):
		keys       = kwargs.keys()
		if ('two_tailed' in keys) and ('dirn' in keys):
			self._warn_two_tailed()
			two_tailed = kwargs['two_tailed']
			dirn       = kwargs['dirn']
			if two_tailed and (dirn!=0):
				raise ValueError('"dirn" can only be 0 when two_tailed=True')
			if (not two_tailed) and (dirn==0):
				raise ValueError('"dirn" must be -1 or 1 when two_tailed=False')
		elif 'two_tailed' in keys:  # and not dirn
			self._warn_two_tailed()
			dirn  = 0 if kwargs['two_tailed'] else 1
		elif 'dirn' in keys:
			dirn  = kwargs['dirn']
		else:
			dirn  = 0  # default value (two_tailed=True)
		if dirn not in [-1, 0, 1]:
			raise ValueError('"dirn" must be -1, 0 or 1')
		return dirn



	def _inference_param(self, alpha, **kwargs):
		
		from . _spm0di import SPM0Di
		
		# preliminary keyword argument checks:
		keys   = kwargs.keys()
		if 'two_tailed' in keys:
			if self.STAT!='T':
				raise ValueError( 'The "two_tailed" option can only be used with T stats.' )

		spmi   = deepcopy( self )
		spmi.__class__ = SPM0Di
	
		if self.STAT == 'T':
			dirn       = self._kwargs2dirn( **kwargs )
			two_tailed = dirn==0
			a          = 0.5*alpha if two_tailed else alpha
			z          = abs(self.z) if two_tailed else (dirn * self.z)
			zc         = stats.t.isf( a, self.df[1] )
			p          = stats.t.sf(  z, self.df[1] )
			p          = min(1, 2*p) if two_tailed else p
		else:
			zc,p   = 0,0
	
		spmi._set_inference_params('param', alpha, zc, p, dirn)
		
		return spmi
		
	
	
	def inference(self, alpha=0.05, method='param', **kwargs):
		
		

		if method=='param':
			# from . _spm0di import SPM0Di
			# spmi   = prob.param_0d(self, alpha, **kwargs)
			spmi   = self._inference_param(alpha, **kwargs)
			# spmi   = SPM0Di(self, alpha, zc, p, two_tailed)
		elif method=='perm':
			spmi   = None 
			# spmi   = prob.perm_0d(self, alpha, **kwargs)
			
			# from .. nonparam.stats import ttest2
			# fro,
			# spm    = ttest2(*self.dv)
			#
			# , **kwargs
			#
			# spmi   = None
			pass
		else:
			raise ValueError('Unknown inference method: {method}. "method" must be one of: ["param", "perm"].')
			# from . nonparam._snpm import SnPM0D_T
			# snpm   = SnPM0D_T(self.z, )
		
		# spmi._set_inference_params(method, alpha, zc, p, two_tailed, dirn)
		
		# spmi._set_testname( self.testname )
		# spmi._set_inference_method( method )
		return spmi





# class _SPM0D(_SPMParent):
# 	def __init__(self, STAT, z, df, beta=None, residuals=None, sigma2=None):
# 		self.STAT           = STAT             # test statistic ("T" or "F")
# 		self.z              = float(z)         # test statistic
# 		self.df             = df               # degrees of freedom
# 		self.beta           = beta             # fitted parameters
# 		self.residuals      = residuals        # model residuals
# 		self.sigma2         = sigma2           # variance
# 		self.testname       = None             # hypothesis test name (set using set_testname method)
#
#
# 	def __repr__(self):
# 		stat     = 't' if self.STAT=='T' else self.STAT
# 		s        = ''
# 		s       += 'SPM{%s} (0D)\n' %stat
# 		s       += '   SPM.testname :  %s\n'      %self.testname
# 		if self.isanova:
# 			s   += '   SPM.effect     :  %s\n'      %self.effect
# 			s   += '   SPM.SS         : (%s, %s)\n' %self.ss
# 			s   += '   SPM.df         : (%s, %s)\n' %self.df
# 			s   += '   SPM.MS         : (%s, %s)\n' %self.ms
# 			s   += '   SPM.z          :  %.5f\n'    %self.z
# 		else:
# 			s   += '   SPM.z        :  %.5f\n'      %self.z
# 			s   += '   SPM.df       :  %s\n'        %dflist2str(self.df)
# 		if self.isregress:
# 			s   += '   SPM.r        :  %.5f\n'      %self.r
# 		s       += '\n'
# 		return s
#
#
#
#
#
# class SPM0D_F(_SPMF, _SPM0D):
# 	def __init__(self, z, df, ss=(0,0), ms=(0,0), eij=0, X0=None):
# 		_SPM0D.__init__(self, 'F', z, df)
# 		self.ss        = tuple( map(float, ss) )
# 		self.ms        = tuple( map(float, ms) )
# 		self.eij       = eij
# 		self.residuals = np.asarray(eij).flatten()
# 		self.X0        = X0
#
# 	def _repr_summ(self):
# 		return '{:<5} F = {:<8} df = {}\n'.format(self.effect_short,  '%.3f'%self.z, dflist2str(self.df))
#
# 	def inference(self, alpha=0.05):
# 		from . _spmi import SPM0Di_F
# 		zstar  = stats.f.isf(alpha, self.df[0], self.df[1])
# 		p      = stats.f.sf(self.z, self.df[0], self.df[1])
# 		return SPM0Di_F(self, alpha, zstar, p)
#
#
#
#
# class SPM0D_T(_SPM0D):
# 	def __init__(self, z, df, beta=None, residuals=None, sigma2=None):
# 		super().__init__('T', z, df, beta=beta, residuals=residuals, sigma2=sigma2)
#
# 	def _inference_param(self, a, z, two_tailed=True):
# 		zc     = stats.t.isf( a, self.df[1] )
# 		p      = stats.t.sf(  z, self.df[1] )
# 		p      = min(1, 2*p) if two_tailed else p
# 		return zc,p
#
# 	def inference(self, alpha=0.05, two_tailed=True, method='param', **kwargs):
# 		a      = 0.5*alpha if two_tailed else alpha
# 		z      = abs(self.z) if two_tailed else self.z
#
# 		if method=='param':
# 			from . _spm0di import SPM0Di_T
# 			zc,p   = self._inference_param(a, z, two_tailed)
# 			spmi   = SPM0Di_T(self, alpha, zc, p, two_tailed)
# 		elif method=='perm':
# 			from .. nonparam.stats import ttest2
# 			fro,
# 			spm    = ttest2(*self.dv)
#
# 			, **kwargs
#
# 			spmi   = None
# 			pass
# 		else:
# 			raise ValueError('Unknown inference method: {method}. "method" must be one of: ["param", "perm"].')
# 			# from . nonparam._snpm import SnPM0D_T
# 			# snpm   = SnPM0D_T(self.z, )
# 		spmi._set_testname( self.testname )
# 		spmi._set_inference_method( method )
# 		return spmi
#
# class SPM0D_T2(_SPM0D):
# 	def __init__(self, z, df):
# 		_SPM0D.__init__(self, 'T2', z, df)
# 	def inference(self, alpha=0.05):
# 		from . _spmi import SPM0Di_T2
# 		zstar  = rft1d.T2.isf0d(alpha, self.df)
# 		p      = rft1d.T2.sf0d( self.z, self.df)
# 		return SPM0Di_T2(self, alpha, zstar, p)
#
# class SPM0D_X2(_SPM0D):
# 	def __init__(self, z, df, residuals=None):
# 		_SPM0D.__init__(self, 'X2', z, df, residuals=residuals)
# 	def inference(self, alpha=0.05):
# 		from . _spmi import SPM0Di_X2
# 		zstar  = rft1d.chi2.isf0d(alpha, self.df[1])
# 		p      = rft1d.chi2.sf0d( self.z, self.df[1])
# 		return SPM0Di_X2(self, alpha, zstar, p)




