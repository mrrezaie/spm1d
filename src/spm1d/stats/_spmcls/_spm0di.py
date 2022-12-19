
'''
SPMi (inference) module

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)

This module contains class definitions for inference SPMs.
'''

# Copyright (C) 2022  Todd Pataky

import warnings
from . _base import _SPMF
from . _spm0d import SPM0D
from ... util import dflist2str



class SPM0Di(SPM0D):
	
	isinference = True
	
	# def __init__(self, spm, alpha, zstar, p, two_tailed=False):
	# 	super().__init__(spm.STAT, spm.z, spm.df, beta=spm.beta, residuals=spm.residuals, sigma2=spm.sigma2)
	# 	self.infmethod   = None           # inference method
	# 	self.alpha       = alpha          # Type I error rate
	# 	self.zstar       = zstar          # critical threshold
	# 	self.h0reject    = abs(spm.z) > zstar if two_tailed else spm.z > zstar
	# 	self.p           = p              # p-value
	# 	self.two_tailed  = two_tailed     # two-tailed test boolean
	# 	self.isanova     = spm.isanova    # ANOVA boolean
	# 	self.isregress   = spm.isregress  # regression boolean
	# 	if self.isanova:
	# 		self.set_effect_label( spm.effect )
	# 		self.ss      = spm.ss
	# 		self.ms      = spm.ms
	# 	if self.isregress:
	# 		self.r       = spm.r

	def __repr__(self):
		s        = ''
		s       += 'SPM{%s} (0D) inference\n'     %self.STAT
		s       += '   SPM.testname         :  %s\n'      %self.testname
		if self.isanova:
			s   += '   SPM.effect           :  %s\n'      %self.effect
			s   += '   SPM.SS               : (%s, %s)\n' %self.ss
			s   += '   SPM.df               : (%s, %s)\n' %self.df
			s   += '   SPM.MS               : (%s, %s)\n' %self.ms
			s   += '   SPM.z                :  %.5f\n'    %self.z
		else:
			s   += '   SPM.z                :  %.5f\n'    %self.z
			s   += '   SPM.df               :  %s\n'      %dflist2str(self.df)
		if self.isregress:
			s   += '   SPM.r                :  %.5f\n'    %self.r
		s       += 'Inference:\n'
		s       += '   SPM.method           :  %s\n'      %self.method
		s       += '   SPM.alpha            :  %.3f\n'    %self.alpha
		s       += '   SPM.dirn             :  %s\n'      %self._dirn_str
		s       += '   SPM.zstar            :  %.5f\n'    %self.zstar
		s       += '   SPM.h0reject         :  %s\n'      %self.h0reject
		s       += '   SPM.p                :  %.5f\n'    %self.p
		s       += '\n'
		return s
	
	@property
	def _dirn_str(self):
		s     = '0 (two-tailed)' if self.dirn==0 else f'{self.dirn} (one-tailed)'
		return s

	@property
	def h0reject(self):
		z,zc  = self.z, self.zc
		return (abs(z) > zc) if self.two_tailed else z > zc
	
	@property
	def isparametric(self):
		return self.method == 'param'
	
	@property
	def two_tailed(self):
		return self.dirn == 0

	@property
	def zstar(self):   # legacy support ("zc" was "zstar" in versions <0.5)
		msg = 'Use of "zstar" is deprecated. Use "zc" to avoid this warning.'
		warnings.warn( msg , DeprecationWarning , stacklevel=2 )
		return self.zc
		



	def _set_inference_params(self, method, alpha, zc, p, dirn):
		self.method      = method         # inference method
		self.alpha       = alpha          # Type I error rate
		self.zc          = zc             # critical value
		self.p           = p              # p-value
		self.dirn        = dirn           # one-tailed direction (-1 or +1)
		# self.isanova     = spm.isanova    # ANOVA boolean
		# self.isregress   = spm.isregress  # regression boolean
		




# class _SPM0Dinference(SPM0D):
#
# 	isinference = True
#
# 	def __init__(self, spm, alpha, zstar, p, two_tailed=False):
# 		_SPM0D.__init__(self, spm.STAT, spm.z, spm.df, beta=spm.beta, residuals=spm.residuals, sigma2=spm.sigma2)
# 		self.infmethod   = None           # inference method
# 		self.alpha       = alpha          # Type I error rate
# 		self.zstar       = zstar          # critical threshold
# 		self.h0reject    = abs(spm.z) > zstar if two_tailed else spm.z > zstar
# 		self.p           = p              # p-value
# 		self.two_tailed  = two_tailed     # two-tailed test boolean
# 		self.isanova     = spm.isanova    # ANOVA boolean
# 		self.isregress   = spm.isregress  # regression boolean
# 		if self.isanova:
# 			self.set_effect_label( spm.effect )
# 			self.ss      = spm.ss
# 			self.ms      = spm.ms
# 		if self.isregress:
# 			self.r       = spm.r
#
# 	def __repr__(self):
# 		s        = ''
# 		s       += 'SPM{%s} (0D) inference\n'     %self.STAT
# 		s       += '   SPM.testname         :  %s\n'      %self.testname
# 		if self.isanova:
# 			s   += '   SPM.effect           :  %s\n'      %self.effect
# 			s   += '   SPM.SS               : (%s, %s)\n' %self.ss
# 			s   += '   SPM.df               : (%s, %s)\n' %self.df
# 			s   += '   SPM.MS               : (%s, %s)\n' %self.ms
# 			s   += '   SPM.z                :  %.5f\n'    %self.z
# 		else:
# 			s   += '   SPM.z                :  %.5f\n'    %self.z
# 			s   += '   SPM.df               :  %s\n'      %dflist2str(self.df)
# 		if self.isregress:
# 			s   += '   SPM.r                :  %.5f\n'    %self.r
# 		s       += 'Inference:\n'
# 		s       += '   SPM.inference_method :  %s\n'      %self.infmethod
# 		s       += '   SPM.alpha            :  %.3f\n'    %self.alpha
# 		s       += '   SPM.zstar            :  %.5f\n'    %self.zstar
# 		s       += '   SPM.h0reject         :  %s\n'      %self.h0reject
# 		s       += '   SPM.p                :  %.5f\n'    %self.p
# 		s       += '\n'
# 		return s
#
# 	@property
# 	def inference_method(self):
# 		return self.infmethod
#
# 	def _set_inference_method(self, method):
# 		self.infmethod = method
#
#
#
#
# class SPM0Di_F(_SPMF, _SPM0Dinference):
# 	'''An SPM{F} (0D) inference object.'''
# 	def _repr_summ(self):
# 		return '{:<5} F = {:<8} df = {:<9} p = {}\n'.format(self.effect.split(' ')[1],  '%.3f'%self.z, dflist2str(self.df), p2string(self.p))
# class SPM0Di_T(_SPM0Dinference):
# 	'''An SPM{T} (0D) inference object.'''
# 	pass
# class SPM0Di_T2(_SPM0Dinference):
# 	'''An SPM{T2} (0D) inference object.'''
# 	pass
# class SPM0Di_X2(_SPM0Dinference):
# 	'''An SPM{X2} (0D) inference object.'''
# 	pass







