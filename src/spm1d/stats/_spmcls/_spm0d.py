
'''
SPM-0D module

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)

This module contains class definitions for 0D SPMs.
'''

# Copyright (C) 2023  Todd Pataky


from copy import deepcopy
import warnings
import numpy as np
from scipy import stats
from . _base import _SPMParent
from ... import prob
from ... util import dflist2str, tuple2str






class SPM0D(_SPMParent):

	def __init__(self, STAT, z, df, beta=None, residuals=None, sigma2=None, X=None):
		self.STAT           = STAT             # test statistic type ("T" or "F")
		self.z              = np.float64(z)    # test statistic value
		self.df             = df               # degrees of freedom
		self.beta           = beta             # fitted parameters
		self.residuals      = residuals        # model residuals
		self.sigma2         = sigma2           # variance
		self.testname       = None             # hypothesis test name (set using set_testname method)
		self.X              = X                # design matrix


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


	def _build_spmi(self, results, alpha, dirn=0, df_adjusted=None):
		from . _spm0di import SPM0Di
		spmi             = deepcopy( self )
		spmi.__class__   = SPM0Di
		spmi.df_adjusted = df_adjusted
		spmi.method      = results.method
		spmi.alpha       = alpha
		spmi.zc          = results.zc
		spmi.p           = results.p
		if self.STAT=='T':
			spmi.dirn     = dirn
			# spmi.dirn   = parser.kwargs['dirn']
		if results.method=='perm':
			spmi.nperm    = results.nperm
			spmi.permuter = results.permuter
		return spmi
		

	def _inference_param(self, alpha, dirn=0, equal_var=False):
		
		df  = self.df
		dfa = None
		
		### heteroscedacity correction:
		if not equal_var:
			from .. import _reml
			yA,yB            = self._args
			y                = np.hstack([yA,yB]) if (self.dim==0) else np.vstack([yA,yB])
			JA,JB            = yA.shape[0], yB.shape[0]
			J                = JA + JB
			q0,q1            = np.eye(JA), np.eye(JB)
			Q0,Q1            = np.matrix(np.zeros((J,J))), np.matrix(np.zeros((J,J)))
			Q0[:JA,:JA]      = q0
			Q1[JA:,JA:]      = q1
			Q                = [Q0, Q1]
			df               = _reml.estimate_df_T(y, self.X, self.residuals, Q)
			dfa              = (1, df)
		
		
		results  = prob.param(self.STAT, self.z, df, alpha=alpha, dirn=dirn)
		return self._build_spmi(results, alpha, dirn=dirn, df_adjusted=dfa)
	

	def _inference_perm(self, alpha, nperm=10000, dirn=0):
		if self.isinlist:
			raise( NotImplementedError( 'Permutation inference must be conducted using the parent SnPMList (for two- and three-way ANOVA).' ) )
		
		results = prob.perm(self.STAT, self.z, alpha=alpha, testname=self.testname, args=self._args, nperm=nperm, dirn=dirn)
		return self._build_spmi(results, alpha, dirn=dirn)
	
	
	def inference(self, alpha, method='param', **kwargs):
		from . _argparsers import InferenceArgumentParser0D
		parser   = InferenceArgumentParser0D(self.STAT, method)
		parser.parse( alpha, **kwargs )

		if method == 'param':
			spmi = self._inference_param(alpha, **kwargs)

		elif method == 'perm':
			spmi = self._inference_perm(alpha, **kwargs)

		return spmi







