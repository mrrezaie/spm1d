
'''
SPMi (inference) module

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)

This module contains class definitions for inference SPMs.
'''

# Copyright (C) 2022  Todd Pataky

import warnings
from . _base import _SPMiParent #, _SPMF
from . _spm0d import SPM0D
from ... util import dflist2str



class SPM0Di(_SPMiParent, SPM0D):
	
	def __repr__(self):
		s        = f'{self._class_str}\n'
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
		s       += '   SPM.isparametric     :  %s\n'      %self.isparametric
		if not self.isparametric:
			s   += '   SPM.nperm_possible   :  %d\n'      %self.nperm_possible
			s   += '   SPM.nperm_actual     :  %d\n'      %self.nperm_actual
		s       += '   SPM.alpha            :  %.3f\n'    %self.alpha
		if self.STAT == 'T':
			s   += '   SPM.dirn             :  %s\n'      %self._dirn_str
		s       += '   SPM.zc               :  %s\n'      %self._zcstr
		s       += '   SPM.h0reject         :  %s\n'      %self.h0reject
		s       += '   SPM.p                :  %.5f\n'    %self.p
		s       += '\n'
		return s
	
	@property
	def h0reject(self):
		z,zc  = self.z, self.zc
		if self.dirn==0:
			zc0,zc1 = (-zc,zc) if self.isparametric else zc
			h       = (z < zc0) or (z > zc1)
		elif self.dirn==1:
			h       = z > zc
		elif self.dirn==-1:
			h       = z < zc
		return h

	@property
	def nperm_actual(self):
		return self.nperm

	@property
	def nperm_possible(self):
		return self.permuter.nPermTotal
	
	@property
	def two_tailed(self):
		return self.dirn == 0

	# def _h0reject_parametric(self):
	# 	z,zc  = self.z, self.zc
	# 	if self.dirn==0:
	# 		h = abs(z) > zc
	# 	elif self.dirn==1:
	# 		h = z > zc
	# 	elif self.dirn==-1:
	# 		h = z < zc
	# 	return h
	#
	# def _h0reject_nonparametric(self):
	# 	z,zc  = self.z, self.zc
	# 	if self.dirn==0:
	# 		zc0,zc1 = zc
	# 		h       = (z < zc0) or (z > zc1)
	# 	elif self.dirn==1:
	# 		h       = z > zc
	# 	elif self.dirn==-1:
	# 		h       = z < zc
	# 	return h




	def _set_inference_params(self, method, alpha, zc, p, dirn):
		self.method      = method         # inference method
		self.alpha       = alpha          # Type I error rate
		self.zc          = zc             # critical value
		self.p           = p              # p-value
		self.dirn        = dirn           # one-tailed direction (-1 or +1)
		# self.isanova     = spm.isanova    # ANOVA boolean
		# self.isregress   = spm.isregress  # regression boolean
		






