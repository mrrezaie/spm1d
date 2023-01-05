
'''
SPM1D class definition
'''

# Copyright (C) 2023  Todd Pataky

from copy import deepcopy
import numpy as np
from . _base import _SPMParent
from ... import prob
from ... plot import plot_spm, plot_spm_design
from ... plot import plot_spmi, plot_spmi_p_values, plot_spmi_threshold_label
from ... util import dflist2str



class SPM1D(_SPMParent):
	dim                     = 1

	def __init__(self, STAT, z, df, beta=None, residuals=None, sigma2=None, X=None, fwhm=None, resels=None, roi=None):
		self.STAT           = STAT             # test statistic ("T" or "F")
		self.X              = X                # design matrix
		self.beta           = beta             # fitted parameters
		self.residuals      = residuals        # residuals (same size as original data)
		self.sigma2         = sigma2           # variance
		self.z              = z                # test statistic
		self.df             = df               # degrees of freedom
		self.fwhm           = fwhm             # smoothness
		self.resels         = resels           # resel counts
		self.roi            = roi              # region of interest

	def __repr__(self):
		s        = f'{self._class_str}\n'
		if self.isanova:
			s   += '   SPM.effect :  %s\n'             %self.effect
		s       += '   SPM.z      :  %s\n'             %self._repr_teststat()
		if self.isregress:
			s   += '   SPM.r      :  %s\n'             %self._repr_corrcoeff()
		s       += '   SPM.df     :  %s\n'             %dflist2str(self.df)
		s       += '   SPM.fwhm   :  %.5f\n'           %self.fwhm
		s       += '   SPM.resels :  (%d, %.5f)\n'     %tuple(self.resels)
		return s
	
	@property
	def R(self):
		return self.residuals
	@property
	def nNodes(self):
		return self.Q
	@property
	def Q(self):
		return self.z.size
	@property
	def Qmasked(self):
		return self.z.size if (self.roi is None) else self.z.count()
	
	
	def _repr_corrcoeff(self):
		return '(1x%d) correlation coefficient field' %self.Q
	def _repr_teststat(self):
		return '(1x%d) test stat field' %self.Q
	def _repr_teststat_short(self):
		return '(1x%d) array' %self.Q


	def _build_spmi(self, results, alpha, dirn=0, df_adjusted=None):
		from . _spm1di import SPM1Di
		spmi             = deepcopy( self )
		spmi.__class__   = SPM1Di
		spmi.df_adjusted = df_adjusted
		spmi.method      = results.method
		spmi.alpha       = alpha
		spmi.zc          = results.zc
		spmi.p_set       = results.p_set
		spmi.clusters    = results.clusters
		if self.STAT=='T':
			spmi.dirn     = dirn
			# spmi.dirn   = parser.kwargs['dirn']
		if results.method=='perm':
			spmi.nperm    = results.nperm
			spmi.permuter = results.permuter
		return spmi


	def inference(self, alpha, method='rft', **kwargs):
		from . _argparsers import InferenceArgumentParser1D
		# parser   = InferenceArgumentParser1D(self.STAT, method)
		# parser.parse( alpha, **kwargs )
		
		if method == 'rft':
			results = prob.rft(self.STAT, self.z, self.df, self.fwhm, self.resels, alpha=alpha, **kwargs)
			# results = self.inference_rft(alpha, **kwargs)
			
			# spmi = self.inference_rft(alpha, **parser.kwargs)
		# elif method == 'perm':
		# 	spmi = self.inference_perm(alpha, **parser.kwargs)
		
		
		# spmi       = self._build_spmi(alpha, zstar, clusters, p_set, two_tailed)    #assemble SPMi object
		# return spmi
		
		dfa = self.df
		dirn = kwargs['dirn']
		
		
		spmi = self._build_spmi(results, alpha, dirn=dirn, df_adjusted=dfa)
		return( spmi )
		
		

		


	def plot(self, **kwdargs):
		return plot_spm(self, **kwdargs)
		
	def plot_design(self, **kwdargs):
		plot_spm_design(self, **kwdargs)
		
	def toarray(self):
		return self.z.copy()





