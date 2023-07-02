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
		self.V    = None   # estimated (co-)variance
		self.X    = None   # design matrix
		self.b    = None   # betas (fitted parameters)
		self.df   = None   # effective degrees of freedom
		self.e    = None   # residuals
		self.f    = None   # F statistic
		self.h    = None   # (co-)variance hyperparameters
		self._v   = None   # variance scale (same as df when equal variance is assumed)

	def __repr__(self):
		s   = 'GeneralLinearModel\n'
		s  += '    X    = %s design matrix\n'      % str(self.X.shape)
		s  += '    C    = %s contrast matrix\n'    % str(self.C.shape)
		s  += '    QQ   = list of %d %s (co-)variance component models\n'  % ( len(self.QQ), f'({self.J},{self.J})' )
		return s
	
	
	@property
	def J(self):
		return None if (self.X is None) else self.X.shape[0]

	# def build_projectors(self):
	# 	pass
	#
	
	def calculate_effective_df(self, X=None):
		# i             = np.any(self.C, axis=1)
		# _X            = self.X[:,i]  # design matrix excluding non-contrast columns
		# _C            = self.C[i]
		X             = self.X if (X is None) else X
		# C             = self.C if (C is None) else C
		trRV,trRVRV   = traceRV(self.V, X)
		trMV,trMVMV   = traceMV(self.V, self.X, self.C)
		df0           = max(trMV**2 / trMVMV, 1.0)
		df1           = trRV**2 / trRVRV
		v0,v1         = trMV, trRV
		self.df       = df0, df1
		self._v       = v0, v1




	# def calculate_effective_df(self):
	# 	# i             = np.any(self.C, axis=1)
	# 	# _X            = self.X[:,i]  # design matrix excluding non-contrast columns
	# 	# _C            = self.C[i]
	#
	# 	# trRV,trRVRV   = traceRV(self.V, self.X)
	# 	trRV,trRVRV   = traceRV(self.V, self.X[:,:-1])
	# 	trMV,trMVMV   = traceMV(self.V, self.X, self.C)
	# 	# trMV,trMVMV   = traceMV(self.V, _X, _C)
	#
	# 	# print('anova1_oop:',  trRV, trRVRV, trMV, trMVMV)
	# 	df0           = max(trMV**2 / trMVMV, 1.0)
	# 	df1           = trRV**2 / trRVRV
	# 	v0,v1         = trMV, trRV
	#
	# 	# print(v1)
	#
	# 	# df1,v1 = 27.0, 27.0
	#
	#
	# 	self.df       = df0, df1
	# 	self._v       = v0, v1


	# def calculate_effective_df(self):
	# 	i             = np.any(self.C, axis=1)
	# 	_X            = self.X[:,i]  # design matrix excluding non-contrast columns
	# 	_C            = self.C[i]
	#
	# 	# trRV,trRVRV   = traceRV(self.V, self.X)
	# 	trRV,trRVRV   = traceRV(self.V, _X)
	# 	trMV,trMVMV   = traceMV(self.V, _X, _C)
	# 	# trMV,trMVMV   = traceMV(self.V, _X, _C)
	#
	# 	# print('anova1_oop:',  trRV, trRVRV, trMV, trMVMV)
	# 	df0           = max(trMV**2 / trMVMV, 1.0)
	# 	df1           = trRV**2 / trRVRV
	# 	v0,v1         = trMV, trRV
	#
	# 	# print(v1)
	#
	# 	# df1,v1 = 27.0, 27.0
	#
	#
	# 	self.df       = df0, df1
	# 	self._v       = v0, v1




	def calculate_f_stat(self):
		# build projectors:
		PX      = self.X @ np.linalg.pinv(self.X)      # X projector
		H       = np.linalg.pinv( self.X.T ) @ self.C  # text between eqns. 9.17 & 9.18 (Friston 2007, p.136)
		PH      = H @ np.linalg.inv(H.T @ H) @ H.T     # H projector
		# f stat:
		YIPY    = self.y.T @ ( np.eye( self.J ) - PX ) @ self.y   # eqn.9.13 denominator (Friston 2007, p.135)
		YPHYY   = self.y.T @ PH @ self.y               # eqn.9.18 (Friston 2007, p.136)
		v0,v1   = self._v
		f       = (YPHYY / v0) / ( YIPY / v1 )         # eqn.9.13 (Friston 2007, p.135)
		self.f  = float(f) if (self.dvdim==1) else np.diag(f)
		
	
	def estimate_variance(self):
		# i             = np.any(self.C, axis=1)
		# _X            = self.X[:,i]  # design matrix excluding non-contrast columns
		# _C            = self.C[i]
		n,s           = self.y.shape
		# trRV          = n - rank(_X)
		trRV          = n - rank(self.X)
		ss            = (self.e**2).sum(axis=0)
		q             = np.diag(  np.sqrt( trRV / ss )  ).T
		Ym            = self.y @ q
		# Ym            = self.e @ q
		YY            = Ym @ Ym.T / s
		V,h           = reml(YY, self.X, self.QQ)
		V            *= (n / np.trace(V))
		self.h        = h
		self.V        = V



	
	def fit(self, y):
		self.dvdim = y.ndim
		self.y = y if (y.ndim==2) else np.array( [y] ).T
		Xi     = np.linalg.pinv( self.X )
		self.b = Xi @ self.y
		self.e = self.y - self.X @ self.b

	
	def greenhouse_geisser(self):
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
		# self.df = (k-1)*e, (n-1)*(k-1)*e    # Eqn.13.36, Friston et al. (2007), p.176

	
	
	def set_contrast_matrix(self, C):
		self.C       = C

	def set_design_matrix(self, X):
		self.X       = X

	def set_variance_model(self, QQ):
		self.QQ      = QQ



