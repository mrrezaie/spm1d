
'''
SPMi (inference) module

(This and all modules whose names start with underscores
are not meant to be accessed directly by the user.)

This module contains class definitions for inference SPMs.
'''

# Copyright (C) 2022  Todd Pataky

from . _spm import _SPM1D




class _SPM1Dinference(_SPM1D):
	'''Parent class for SPM inference objects.'''
	
	isinference = True
	
	def __init__(self, spm, alpha, zstar, clusters, p_set, p, two_tailed=False):
		_SPM.__init__(self, spm.STAT, spm.z, spm.df, spm.fwhm, spm.resels, X=spm.X, beta=spm.beta, residuals=spm.residuals, sigma2=spm.sigma2, roi=spm.roi)
		self.alpha       = alpha               #Type I error rate
		self.zstar       = zstar               #critical threshold
		self.clusters    = clusters            #supra-threshold cluster information
		self.isregress   = spm.isregress       #propagate regression flag
		self.nClusters   = len(clusters)       #number of supra-threshold clusters
		self.h0reject    = self.nClusters > 0  #null hypothesis rejection decision
		self.p_set       = p_set               #set-level p value
		self.p           = p                   #cluster-level p values
		self.two_tailed  = two_tailed          #two-tailed test boolean
		if self.isregress:
			self.r       = spm.r
		if self.isanova:
			self.set_effect_label( spm.effect )

	def __repr__(self):
		stat     = 't' if self.STAT == 'T' else self.STAT
		s        = ''
		s       += 'SPM{%s} inference field\n' %stat
		if self.isanova:
			s   += '   SPM.effect    :   %s\n' %self.effect
		s       += '   SPM.z         :  (1x%d) raw test stat field\n' %self.Q
		if self.isregress:
			s   += '   SPM.r         :  %s\n' %self._repr_corrcoeff()
		s       += '   SPM.df        :  %s\n' %dflist2str(self.df)
		s       += '   SPM.fwhm      :  %.5f\n' %self.fwhm
		s       += '   SPM.resels    :  (%d, %.5f)\n' %tuple(self.resels)
		s       += 'Inference:\n'
		s       += '   SPM.alpha     :  %.3f\n' %self.alpha
		s       += '   SPM.zstar     :  %.5f\n' %self.zstar
		s       += '   SPM.h0reject  :  %s\n' %self.h0reject
		s       += '   SPM.p_set     :  %s\n' %p2string(self.p_set)
		s       += '   SPM.p_cluster :  (%s)\n\n\n' %plist2string(self.p)
		return s
	
	def plot(self, **kwdargs):
		return plot_spmi(self, **kwdargs)

	def plot_p_values(self, **kwdargs):
		plot_spmi_p_values(self, **kwdargs)
	
	def plot_threshold_label(self, **kwdargs):
		return plot_spmi_threshold_label(self, **kwdargs)
	
	





class SPM1Di_T(_SPM1Dinference):
	'''An SPM{t} inference continuum.'''
	pass
class SPM1Di_F(_SPMF, _SPM1Dinference):
	'''An SPM{F} inference continuum.'''
	def _repr_summ(self):
		return '{:<5} z={:<18} df={:<9} h0reject={}\n'.format(self.effect_short,  self._repr_teststat_short(), dflist2str(self.df), self.h0reject)
	
class SPM1Di_T2(_SPM1Dinference):
	'''An SPM{T2} inference continuum.'''
	pass
class SPM1Di_X2(_SPM1Dinference):
	'''An SPM{X2} inference continuum.'''
	pass







