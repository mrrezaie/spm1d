
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
from .. import prob
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


	def _isf_sf_t(self, a, z, df, dirn=0):
		_z     = np.abs(z) if (dirn==0) else dirn*z
		a      = 0.5 * a if (dirn==0) else a
		zc     = stats.t.isf( a, df )
		zc     = -zc if (dirn==-1) else zc
		p      = stats.t.sf( _z, df )
		p      = min(1, 2*p) if (dirn==0) else p
		return zc,p

	def _inference_gauss_t(self, alpha, dirn=0):
		from . _spm0di import SPM0Di
		zc,p           = self._isf_sf_t(alpha, self.z, self.df[1], dirn)
		spmi           = deepcopy( self )
		spmi.__class__ = SPM0Di
		spmi.method    = 'gauss'
		spmi.alpha     = alpha
		spmi.zc        = zc
		spmi.p         = p
		spmi.dirn      = dirn
		return spmi
		

	def _inference_gauss(self, alpha):
		pass
	
	

	def _inference_perm_t(self, alpha, dirn=0, nperm=10000):
		# two_tailed = dirn==0

		from .. nonparam.permuters import get_permuter
		permuter   = get_permuter(self.testname, self.dim)( *self._args )
		
		# print(permuter)
		# print(permuter.nPermTotal)
		
		if nperm > permuter.nPermTotal:
			raise ValueError( f'Number of specified permutations ({nperm}) exceeds the total number of possible permutations ({permuter.nPermTotal}).')

		permuter.build_pdf(  nperm  )
		zc         = permuter.get_z_critical(alpha, dirn)



		if not isinstance(zc, float):
			zc     = zc.flatten()
		p          = permuter.get_p_value(self.z, zc, alpha, dirn)

		from . _spm0di import SPM0Di
		spmi             = deepcopy( self )
		spmi.__class__   = SPM0Di
		spmi.method      = 'perm'
		spmi.alpha       = alpha
		spmi.zc          = zc
		spmi.p           = p
		spmi.dirn        = dirn
		spmi.nperm       = nperm
		spmi.permuter    = permuter
		# spmi._set_inference_params('perm', alpha, zc, p, dirn)
		return spmi


		
	
	def _inference_perm(self, alpha, nperm=-1):
		pass
	
	
	
	
	
	def inference(self, alpha, method='gauss', **kwargs):
		parser   = InferenceArgumentParser0D(self.STAT, method)
		parser.parse( alpha, **kwargs )
		
		if method == 'gauss':
			spmi = self.inference_gauss(alpha, **parser.kwargs)
		elif method == 'perm':
			spmi = self.inference_perm(alpha, **parser.kwargs)
		return spmi
		
		
	
	def inference_gauss(self, alpha, **kwargs):
		f = self._inference_gauss_t if (self.STAT=='T') else self._inference_gauss
		return f(alpha, **kwargs)


	def inference_perm(self, alpha, **kwargs):
		if self.isinlist:
			raise( NotImplementedError( 'Non-parametric inference must be conducted using the parent SnPMList (for two- and three-way ANOVA).' ) )
		# self._check_iterations(iterations, alpha, force_iterations)
		f = self._inference_perm_t if (self.STAT=='T') else self._inference_perm
		return f(alpha, **kwargs)





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




