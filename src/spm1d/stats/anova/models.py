'''
ANOVA
'''

# Copyright (C) 2023  Todd Pataky



import numpy as np
import scipy.stats
import rft1d
from .. _cov import reml, traceRV, traceMV, _reml_old
from .. _la import rank
from ... util.str import array2shortstr



class GeneralLinearModel(object):
	def __init__(self):
		self.C    = None   # contrast matrix
		self.QQ   = None   # (co-)variance model
		self.X    = None   # design matrix

	def __repr__(self):
		s   = 'GeneralLinearModel\n'
		s  += '    X    = %s design matrix\n'      % str(self.X.shape)
		s  += '    C    = %s contrast matrix\n'    % str(self.C.shape)
		s  += '    QQ   = list of %d %s (co-)variance component models\n'  % ( len(self.QQ), f'({self.J},{self.J})' )
		return s
	
	@property
	def J(self):
		return None if (self.X is None) else self.X.shape[0]

	def fit(self, y):
		y      = np.asarray(y, dtype=float)
		y      = y if (y.ndim==2) else np.array([y]).T
		Xi     = np.linalg.pinv( self.X )
		b      = Xi @ y
		e      = y - self.X @ b
		return FittedGLM(self, y, b, e)


	def set_contrast_matrix(self, C):
		self.C       = C
	def set_design_matrix(self, X):
		self.X       = X
	def set_variance_model(self, QQ):
		self.QQ      = QQ



class FittedGLM(object):
	def __init__(self, model, y, b, e):
		self._df   = None   # effective degrees of freedom (for one contrast)
		self._f    = None   # F statistic (for one contrast)
		self._v    = None   # variance scale (for one contrast); same as df when equal variance is assumed
		self.model = model
		self.V     = None   # estimated (co-)variance
		self.b     = b      # betas (fitted parameters)
		self.dvdim = 1 if ((y.ndim==1) or (1 in y.shape)) else y.ndim # dependent variable dimensionality
		self.e     = e      # residuals
		self.h     = None   # (co-)variance hyperparameters
		self.y     = y      # (J,Q) dependent variable array where J=num.observations and Q=num.continuum nodes
		
		

	@property
	def J(self):
		return self.model.J

	
	def adjust_df_greenhouse_geisser(self):
		pass
		#
		# # this works for 0D data:
		# if self.dvdim==1:
		# 	y      = self.y.ravel()
		# 	ytable = np.vstack(  [y[self.A==u]  for u in self.uA]  )
		# 	k,n    = ytable.shape
		# 	S      = np.cov(ytable, ddof=1)
		# else:
		# 	S       = np.array([h*q  for h,q in zip(self.h, self.QQ)]).sum(axis=0)  # Eqn.10.14  (Friston, 2007)
		# 	k       = self.uA.size
		# 	n       = self.J / k
		# # # my BAD 1D attempt:
		# # k      = self.uA.size
		# # n      = self.J / k
		# # S      = self.V
		#
		# # # new 1D attempt:
		# # S       = np.array([h*q  for h,q in zip(self.h, self.QQ)]).sum(axis=0)  # Eqn.10.14  (Friston, 2007)
		# # k       = self.uA.size
		# # n       = self.J / k
		# # eps    = _greenhouse_geisser( S, k )
		#
		# w,v     = np.linalg.eig( S )
		# ind     = w.argsort()[::-1]
		# v       = v[:,ind]
		# M       = v[:,:k-1]
		# Sz      = M.T @ S @ M
		# w,_     = np.linalg.eig( Sz )
		# e       = w.sum()**2 / (   (k-1) * (w**2).sum()   )   # Eqn.13.37, Friston et al. (2007), p.176
		# self._df = (k-1)*e, (n-1)*(k-1)*e    # Eqn.13.36, Friston et al. (2007), p.176

	def calculate_effective_df(self, X=None):
		X             = self.model.X if (X is None) else X
		trRV,trRVRV   = traceRV(self.V, X)
		trMV,trMVMV   = traceMV(self.V, self.model.X, self.model.C)
		df0           = max(trMV**2 / trMVMV, 1.0)
		df1           = trRV**2 / trRVRV
		v0,v1         = trMV, trRV
		self._df       = df0, df1
		self._v       = v0, v1

	def calculate_f_stat(self):
		# build projectors:
		y,X,C   = self.y, self.model.X, self.model.C
		PX      = X @ np.linalg.pinv(X)      # X projector
		H       = np.linalg.pinv( X.T ) @ C  # text between eqns. 9.17 & 9.18 (Friston 2007, p.136)
		PH      = H @ np.linalg.inv(H.T @ H) @ H.T     # H projector
		# f stat:
		YIPY    = y.T @ ( np.eye( self.J ) - PX ) @ y   # eqn.9.13 denominator (Friston 2007, p.135)
		YPHYY   = y.T @ PH @ y               # eqn.9.18 (Friston 2007, p.136)
		v0,v1   = self._v
		f       = (YPHYY / v0) / ( YIPY / v1 )         # eqn.9.13 (Friston 2007, p.135)
		self._f  = float(f) if (self.dvdim==1) else np.diag(f)
		
	
	def estimate_variance(self):
		# i             = np.any(self.C, axis=1)
		# _X            = self.X[:,i]  # design matrix excluding non-contrast columns
		# _C            = self.C[i]
		# X             = self.model.X
		n,s           = self.y.shape
		# trRV          = n - rank(_X)
		trRV          = n - rank(self.model.X)
		ss            = (self.e**2).sum(axis=0)
		q             = np.diag(  np.sqrt( trRV / ss )  ).T
		Ym            = self.y @ q
		# Ym            = self.e @ q
		YY            = Ym @ Ym.T / s
		V,h           = reml(YY, self.model.X, self.model.QQ)
		V            *= (n / np.trace(V))
		self.h        = h
		self.V        = V