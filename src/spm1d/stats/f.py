'''
ANOVA
'''

# Copyright (C) 2023  Todd Pataky



import numpy as np
import scipy.stats
import rft1d
from . _cov import reml, traceRV, traceMV
from . _la import rank



class OneWayANOVAModel(object):
	def __init__(self, y, A, *args, Q=None):
		self.ndim = y.ndim
		self.y    = y if (self.ndim==2) else np.array( [y] ).T
		self.A    = np.asarray(A, dtype=float).flatten()
		self.uA   = np.unique( self.A )
		self.X    = None   # design matrix
		self.C    = None   # contrast matrix
		self.df   = None   # effective degrees of freedom
		self.e    = None   # residuals
		self.f    = None   # F statistic
		self._v   = None   # variance scale (same as df when equal variance is assumed)
		self._build_design_matrix()
		self._build_contrast_matrix()

	@property
	def J(self):
		return self.A.size


	# def _build_contrast_matrix(self):
	# 	n        = self.X.shape[1]
	# 	C        = np.zeros( (n-1, n) )
	# 	for i in range(n-1):
	# 		C[i,i]   = 1
	# 		C[i,i+1] = -1
	# 	self.C   = C.T
	#
	# def _build_design_matrix(self):
	# 	u        = self.uA
	# 	JJ       = [(self.A==uu).sum()  for uu in u]
	# 	n        = u.size
	# 	X        = np.zeros( (self.J,n) )
	# 	for i,uu in enumerate(u):
	# 		X[:,i]  = self.A==uu
	# 	self.X   = X

	def _build_contrast_matrix(self):
		n        = self.X.shape[1]
		C        = np.zeros( (n-2, n) )
		for i in range(n-2):
			C[i,i]   = 1
			C[i,i+1] = -1
		self.C   = C.T

	def _build_design_matrix(self):
		u        = self.uA
		# JJ       = [(self.A==uu).sum()  for uu in u]
		n        = u.size
		X        = np.zeros( (self.J,n+1) )
		for i,uu in enumerate(u):
			X[:,i]  = self.A==uu
		X[:,n]   = 1
		self.X   = X


	# def build_projectors(self):
	# 	pass
	#
	
	def calculate_effective_df(self):
		i             = np.any(self.C, axis=1)
		_X            = self.X[:,i]  # design matrix excluding non-contrast columns
		_C            = self.C[i]
		
		# print( _X.shape )
		
		
		trRV,trRVRV   = traceRV(self.V, self.X[:,:-1])
		trMV,trMVMV   = traceMV(self.V, self.X, self.C)
		
		
		# trRV,trRVRV   = traceRV(self.V, _X)
		# trMV,trMVMV   = traceMV(self.V, _X, _C)
		# print('anova1_oop:',  trRV, trRVRV, trMV, trMVMV)
		df0           = max(trMV**2 / trMVMV, 1.0)
		df1           = trRV**2 / trRVRV
		v0,v1         = trMV, trRV
		self.df       = df0, df1
		self._v       = v0, v1
		
	
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
		self.f  = float(f) if (self.ndim==1) else np.diag(f)
		
	
	def estimate_variance(self, equal_var=False):
		i             = np.any(self.C, axis=1)
		_X            = self.X[:,i]  # design matrix excluding non-contrast columns
		_C            = self.C[i]
		if equal_var:
			Q  = [np.eye(self.J)]
		else:
			Q  = [np.asarray(np.diag( self.A==uu ), dtype=float)  for uu in self.uA]
		n,s           = self.y.shape
		trRV          = n - rank(_X)
		ss            = (self.e**2).sum(axis=0)
		q             = np.diag(  np.sqrt( trRV / ss )  ).T
		Ym            = self.y @ q
		YY            = Ym @ Ym.T / s
		V,h           = reml(YY, _X, Q)
		V            *= (n / np.trace(V))
		self.V        = V

	# def estimate_variance(self, equal_var=False):
	# 	if equal_var:
	# 		Q  = [np.eye(self.J)]
	# 	else:
	# 		Q  = [np.asarray(np.diag( self.A==uu ), dtype=float)  for uu in self.uA]
	# 	n,s           = self.y.shape
	# 	trRV          = n - rank(self.X)
	# 	ss            = (self.e**2).sum(axis=0)
	# 	q             = np.diag(  np.sqrt( trRV / ss )  ).T
	# 	Ym            = self.y @ q
	# 	YY            = Ym @ Ym.T / s
	# 	V,h           = reml(YY, self.X, Q)
	# 	V            *= (n / np.trace(V))
	# 	self.V        = V
				
	def fit(self):
		Xi     = np.linalg.pinv( self.X )
		b      = Xi @ self.y
		self.e = self.y - self.X @ b   # residuals


def anova1_oop(y, A, equal_var=False):
	model  = OneWayANOVAModel(y, A)
	model.fit()
	model.estimate_variance( equal_var=equal_var )
	model.calculate_effective_df()
	model.calculate_f_stat()
	f,df   = model.f, model.df
	
	if isinstance(f, float):
		p  = scipy.stats.f.sf(f, df[0], df[1])
	else:
		fwhm    = rft1d.geom.estimate_fwhm( model.e )
		p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
	
	return f, df, p








def anova1(y, A, equal_var=False):
	ndim    = y.ndim
	J       = y.shape[0]
	u       = np.unique(A)
	JJ      = [(A==uu).sum()  for uu in u]
	nf      = u.size
	# # design matrix
	# X       = np.zeros( (J,nf) )
	# X[:,0]  = 1
	# for i,uu in enumerate(u[1:]):
	# 	X[:,i+1]  = A==uu
	# design matrix
	X       = np.zeros( (J,nf) )
	for i,uu in enumerate(u):
		X[:,i]  = A==uu
	# reduced design matrix
	Xr      = np.ones( (J,1) )
	Y       = y if (ndim==2) else np.array( [y] ).T
	# contrast matrix:
	c             = np.zeros( (nf-1, nf) )
	for i in range(nf-1):
		c[i,i]   = 1
		c[i,i+1] = -1
	c = c.T
	
	# residuals:
	Xi   = np.linalg.pinv(X)
	b    = Xi @ Y
	eij  = Y - X @ b
	# covariance model:
	if equal_var:
		Q  = [np.eye(J)]
	else:
		Q  = [np.asarray(np.diag( A==uu ), dtype=float)  for uu in u]
	# estimate covariance and DF
	n,s           = Y.shape
	trRV          = n - rank(X)
	ResSS         = (np.asarray(eij)**2).sum(axis=0)
	q             = np.diag(np.sqrt( trRV / ResSS )).T
	Ym            = Y @ q
	YY            = Ym @ Ym.T / s
	V,h           = reml(YY, X, Q)
	V            *= (n / np.trace(V))
	### effective degrees of freedom (denominator):
	trRV,trRVRV   = traceRV(V, X)
	df1           = trRV**2 / trRVRV
	### effective degrees of freedom (numerator):
	trMV,trMVMV   = traceMV(V, X, c)
	# print('anova1:  f=%0.3f, df=(%0.3f,f=%0.3f), p=%0.5f', )
	# print('anova1:',  trRV, trRVRV, trMV, trMVMV)
	df0           =  trMV**2 / trMVMV
	df0           = max(df0,1.0)
	v0,v1         = trMV, trRV
	
	# F stat
	I       = np.eye(J)
	P       = X @ np.linalg.inv(X.T @ X) @ X.T
	Pr      = Xr @ np.linalg.inv(Xr.T @ Xr) @ Xr.T
	YIPY    = Y.T @ ( I - P ) @ Y
	YIPrY   = Y.T @ ( I - Pr ) @ Y
	f       = ((YIPrY - YIPY) / v0) / ( YIPY / v1 )
	f       = float(f) if (ndim==1) else np.diag(f)
	
	if ndim==1:
		p       = scipy.stats.f.sf(f, df0, df1)
	else:
		Q       = y.shape[1]
		efwhm   = rft1d.geom.estimate_fwhm( eij )
		p       = rft1d.f.sf(f.max(), (df0, df1), Q, efwhm)

	return f, (df0,df1), V, p
	
	# return f, None, None, None


