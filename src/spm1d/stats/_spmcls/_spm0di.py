
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
	
	isinference   = True
	
	def __repr__(self):
		s          = super().__repr__()
		s         += 'Inference:\n'
		s         += '   SPM.method            :  %s\n'      %self.method
		s         += '   SPM.isparametric      :  %s\n'      %self.isparametric
		if self.ismultigroup:
			a    = self.eqvar_assumed
			s     += '   SPM.eqvar_assumed     :  %s\n' %a
			if not a:
				s += '   SPM.df_adjusted       :  %s\n' %dflist2str(self.df_adjusted)
			
		if not self.isparametric:
			s     += '   SPM.nperm_possible    :  %d\n'      %self.nperm_possible
			s     += '   SPM.nperm_actual      :  %d\n'      %self.nperm_actual
		s         += '   SPM.alpha             :  %.3f\n'    %self.alpha
		if self.STAT == 'T':
			s     += '   SPM.dirn              :  %s\n'      %self._dirn_str
		s         += '   SPM.zc                :  %s\n'      %self._zcstr
		s         += '   SPM.h0reject          :  %s\n'      %self.h0reject
		s         += '   SPM.p                 :  %.5f\n'    %self.p
		s         += '\n'
		return s
	
	@property
	def h0reject(self):
		z,zc  = self.z, self.zc
		if self.dirn==0:
			h = (z < -zc) or (z > zc)
		elif self.dirn==1:
			h = z > zc
		elif self.dirn==-1:
			h = z < -zc
		return h

	# @property
	# def h0reject(self):
	# 	z,zc  = self.z, self.zc
	# 	if self.dirn in is None:
	# 		h       = z > zc
	# 	# elif self.dirn==0:
	# 	# 	zc0,zc1 = (-zc,zc) if self.isparametric else zc
	# 	# 	h       = (z < zc0) or (z > zc1)
	# 	elif self.dirn==0:
	# 		h       = (z < -zc) or (z > zc)
	# 	elif self.dirn==1:
	# 		h       = z > zc
	# 	elif self.dirn==-1:
	# 		h       = z < -zc
	# 	return h

	# @property
	# def eqvar_assumed(self):
	# 	return self.df_adjusted is None
	#
	# @property
	# def nperm_actual(self):
	# 	return self.nperm
	#
	# @property
	# def nperm_possible(self):
	# 	return self.permuter.nPermTotal
	#
	# @property
	# def two_tailed(self):
	# 	return self.dirn == 0






