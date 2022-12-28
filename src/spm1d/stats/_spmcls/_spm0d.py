
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
from . _argparsers import InferenceArgumentParser0D, InferenceArgumentParser1D
# from .. _argparse import KeywordArgumentParser
# from .. import prob
from ... import prob
from ... util import dflist2str
from . _base import _SPMParent





class SPM0D(_SPMParent):

	def __init__(self, STAT, z, df, beta=None, residuals=None, sigma2=None):
		self.STAT           = STAT             # test statistic type ("T" or "F")
		self.z              = np.float64(z)    # test statistic value
		self.df             = df               # degrees of freedom
		self.beta           = beta             # fitted parameters
		self.residuals      = residuals        # model residuals
		self.sigma2         = sigma2           # variance
		self.testname       = None             # hypothesis test name (set using set_testname method)


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




	def inference_param(self, alpha, dirn=0):
		from . _spm0di import SPM0Di
		
		results        = prob.param(self.STAT, self.z, self.df, alpha=alpha, dirn=dirn)
		# spmi = self.inference_gauss(alpha, **parser.kwargs)
		
		spmi           = deepcopy( self )
		spmi.__class__ = SPM0Di
		spmi.method    = 'param'
		spmi.alpha     = alpha
		spmi.zc        = results.zc
		spmi.p         = results.p
		if self.STAT=='T':
			spmi.dirn  = dirn
			# spmi.dirn  = parser.kwargs['dirn']
		return spmi
	

	def inference_perm(self, alpha, nperm=10000, dirn=0):
		
		if self.isinlist:
			raise( NotImplementedError( 'Permutation inference must be conducted using the parent SnPMList (for two- and three-way ANOVA).' ) )
		
		# self._check_iterations(iterations, alpha, force_iterations)
		
		from . _spm0di import SPM0Di
		
		results          = prob.perm(self.STAT, self.z, alpha=alpha, testname=self.testname, args=self._args, nperm=nperm, dirn=dirn)
		
		spmi             = deepcopy( self )
		spmi.__class__   = SPM0Di
		spmi.method      = 'perm'
		spmi.alpha       = alpha
		spmi.zc          = results.zc
		spmi.p           = results.p
		spmi.nperm       = nperm
		spmi.permuter    = results.permuter
		if self.STAT=='T':
			# spmi.dirn    = parser.kwargs['dirn']
			spmi.dirn    = dirn
			
		return spmi
	
	
	
	
	
	def inference(self, alpha, method='param', **kwargs):
		parser   = InferenceArgumentParser0D(self.STAT, method)
		parser.parse( alpha, **kwargs )

		if method == 'param':
			spmi = self.inference_param(alpha, **kwargs)

		elif method == 'perm':
			spmi = self.inference_perm(alpha, **kwargs)

		return spmi




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




