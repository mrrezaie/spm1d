
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
from ... util import dflist2str, tuple2str
from . _base import _SPM0DParent





class SPM0D(_SPM0DParent):

	def __init__(self, STAT, z, df, beta=None, residuals=None, sigma2=None):
		self.STAT           = STAT             # test statistic type ("T" or "F")
		self.z              = np.float64(z)    # test statistic value
		self.df             = df               # degrees of freedom
		self.beta           = beta             # fitted parameters
		self.residuals      = residuals        # model residuals
		self.sigma2         = sigma2           # variance
		self.testname       = None             # hypothesis test name (set using set_testname method)


	def __repr__(self):
		s        = f'{self._class_str}\n'
		s       += '   SPM.testname         :  %s\n'        %self.testname
		if self.isanova:
			s   += '   SPM.effect_label     :  %s\n'        %self.effect_label
			s   += '   SPM.ms               :  %s\n'        %tuple2str(self.ms, '%.3f')
			s   += '   SPM.ss               :  %s\n'        %tuple2str(self.ss, '%.3f')
		s       += '   SPM.z                :  %.5f\n'      %self.z
		s       += '   SPM.df               :  %s\n'        %dflist2str(self.df)
		if self.isregress:
			s   += '   SPM.r                :  %.5f\n'      %self.r
		s       += '\n'
		return s


	def _build_spmi(self, results, alpha, dirn=0):
		from . _spm0di import SPM0Di
		spmi           = deepcopy( self )
		spmi.__class__ = SPM0Di
		spmi.method    = results.method
		spmi.alpha     = alpha
		spmi.zc        = results.zc
		spmi.p         = results.p
		if self.STAT=='T':
			spmi.dirn     = dirn
			# spmi.dirn   = parser.kwargs['dirn']
		if results.method=='perm':
			spmi.nperm    = results.nperm
			spmi.permuter = results.permuter
		return spmi
		

	def _inference_param(self, alpha, dirn=0):
		results  = prob.param(self.STAT, self.z, self.df, alpha=alpha, dirn=dirn)
		return self._build_spmi(results, alpha, dirn=dirn)
	

	def _inference_perm(self, alpha, nperm=10000, dirn=0):
		if self.isinlist:
			raise( NotImplementedError( 'Permutation inference must be conducted using the parent SnPMList (for two- and three-way ANOVA).' ) )
		
		results = prob.perm(self.STAT, self.z, alpha=alpha, testname=self.testname, args=self._args, nperm=nperm, dirn=dirn)
		return self._build_spmi(results, alpha, dirn=dirn)
		
	
	
	
		# # self._check_iterations(iterations, alpha, force_iterations)
		#
		# from . _spm0di import SPM0Di
		#
		# results          = prob.perm(self.STAT, self.z, alpha=alpha, testname=self.testname, args=self._args, nperm=nperm, dirn=dirn)
		#
		# spmi             = deepcopy( self )
		# spmi.__class__   = SPM0Di
		# spmi.method      = 'perm'
		# spmi.alpha       = alpha
		# spmi.zc          = results.zc
		# spmi.p           = results.p
		# spmi.nperm       = nperm
		# spmi.permuter    = results.permuter
		# if self.STAT=='T':
		# 	# spmi.dirn    = parser.kwargs['dirn']
		# 	spmi.dirn    = dirn
		#
		# return spmi
		#
	
	
	
	
	def inference(self, alpha, method='param', **kwargs):
		parser   = InferenceArgumentParser0D(self.STAT, method)
		parser.parse( alpha, **kwargs )

		if method == 'param':
			spmi = self._inference_param(alpha, **kwargs)

		elif method == 'perm':
			spmi = self._inference_perm(alpha, **kwargs)

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




