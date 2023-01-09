
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
from ... util import array2shortstr, arraytuple2str, dflist2str, resels2str, DisplayParams



class SPM1D(_SPMParent):
	dim                     = 1

	def __init__(self, STAT, z, df, beta=None, residuals=None, sigma2=None, X=None, fwhm=None, resels=None, roi=None):
		self.STAT           = STAT             # test statistic ("T" or "F")
		self.testname       = None             # hypothesis test name (set using set_testname method)
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
		dp      = DisplayParams( self )
		dp.add_header( self._class_str )
		dp.add( 'testname' )
		dp.add( 'STAT' )
		if self.isanova:
			dp.add( 'effect_label' )
			dp.add( 'ss' , arraytuple2str )
			dp.add( 'ms' , arraytuple2str )
		dp.add( 'z', fmt=array2shortstr )
		if self.isregress:
			dp.add('r', fmt=array2shortstr )
		dp.add( 'df', fmt=dflist2str )
		dp.add( 'fwhm', fmt='%.3f' )
		dp.add( 'resels', fmt=resels2str )
		return dp.asstr()

	
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
	def _repr_summ(self):  # for F lists
		return '{:<5} z = {:<18} df = {}\n'.format(self.effect_label_s,  self._repr_teststat_short(), dflist2str(self.df))


	# def _build_spmi(self, results, alpha, dirn=0, df_adjusted=None):
	# 	from . _spm1di import SPM1Di
	# 	spmi             = SPM1Di( self, results.method, alpha, results.zc, results.p, df_adjusted, dirn )
	#
	# 	spmi             = deepcopy( self )
	# 	spmi.__class__   = SPM1Di
	# 	spmi.df_adjusted = df_adjusted
	# 	spmi.method      = results.method
	# 	spmi.alpha       = alpha
	# 	spmi.zc          = results.zc
	# 	spmi.p_set       = results.p_set
	# 	spmi.p_max       = results.p_max
	# 	spmi.clusters    = results.clusters
	# 	if self.STAT=='T':
	# 		spmi.dirn     = dirn
	# 		# spmi.dirn   = parser.kwargs['dirn']
	# 	if results.method=='perm':
	# 		spmi.nperm    = results.nperm
	# 		spmi.permuter = results.permuter
	# 	return spmi


	def _build_spmi(self, results, df_adjusted=None):
		from . _spm1di import SPM1Di
		# spmi             = deepcopy( self )
		# spmi.__class__   = SPM0Di
		spmi             = SPM1Di(self, results, df_adjusted)
		# if results.method=='perm':
		# 	spmi.nperm    = results.nperm
		# 	spmi.permuter = results.permuter
		return spmi

	# def _build_spmi(self, results, alpha, dirn=0, df_adjusted=None):
	# 	from . _spm1di import SPM1Di
	# 	spmi             = deepcopy( self )
	# 	spmi.__class__   = SPM1Di
	# 	spmi.df_adjusted = df_adjusted
	# 	spmi.method      = results.method
	# 	spmi.alpha       = alpha
	# 	spmi.zc          = results.zc
	# 	spmi.p_set       = results.p_set
	# 	spmi.p_max       = results.p_max
	# 	spmi.clusters    = results.clusters
	# 	if self.STAT=='T':
	# 		spmi.dirn     = dirn
	# 		# spmi.dirn   = parser.kwargs['dirn']
	# 	if results.method=='perm':
	# 		spmi.nperm    = results.nperm
	# 		spmi.permuter = results.permuter
	# 	return spmi


	def inference(self, alpha, method='rft', **kwargs):
		# from . _argparsers import InferenceArgumentParser1D
		# parser   = InferenceArgumentParser1D(self.STAT, self.testname, method)
		# kwargs   = parser.parse( alpha, **kwargs )
		
		if method == 'rft':
			results = prob.rft(self.STAT, self.z, self.df, self.fwhm, self.resels, alpha=alpha, **kwargs)
			# results = self.inference_rft(alpha, **kwargs)
			
			# spmi = self.inference_rft(alpha, **parser.kwargs)
		elif method == 'perm':
			# print( len(self._args) )
			# nperm = kwargs['nperm']
			# results = prob.perm(self.STAT, self.z, alpha=alpha, testname=self.testname, args=self._args, nperm=nperm, dirn=dirn)
			results = prob.perm(self.STAT, self.z, alpha=alpha, testname=self.testname, args=self._args, dim=1, **kwargs)
			
			# return self._build_spmi(results, alpha, dirn=dirn)
			#
			# spmi = self.inference_perm(alpha, **parser.kwargs)
		
		
		# spmi       = self._build_spmi(alpha, zstar, clusters, p_set, two_tailed)    #assemble SPMi object
		# return spmi
		
		dfa = self.df
		# dirn = kwargs['dirn'] if 'dirn' in kwargs else None
		
		
		spmi = self._build_spmi(results, df_adjusted=dfa)
		return( spmi )
		
		

		


	def plot(self, **kwdargs):
		return plot_spm(self, **kwdargs)
		
	def plot_design(self, **kwdargs):
		plot_spm_design(self, **kwdargs)
		
	def toarray(self):
		return self.z.copy()





