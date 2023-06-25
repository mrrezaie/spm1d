'''
ANOVA
'''

# Copyright (C) 2023  Todd Pataky



import numpy as np
import scipy.stats
import rft1d
from .. _cov import reml, traceRV, traceMV, _reml_old
from .. _la import rank

class GLM(object):
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
		pass

	def _build_design_matrix(self):
		pass


	# def build_projectors(self):
	# 	pass
	#
	
	

	
	# def calculate_effective_df(self):
	# 	# i             = np.any(self.C, axis=1)
	# 	# _X            = self.X[:,i]  # design matrix excluding non-contrast columns
	# 	# _C            = self.C[i]
	# 	trRV,trRVRV   = traceRV(self.V, self.X[:,:-1])
	# 	trMV,trMVMV   = traceMV(self.V, self.X, self.C)
	# 	df0           = max(trMV**2 / trMVMV, 1.0)
	# 	df1           = trRV**2 / trRVRV
	# 	v0,v1         = trMV, trRV
	# 	self.df       = df0, df1
	# 	self._v       = v0, v1
		
	
	# def calculate_effective_df(self):
	# 	i             = np.any(self.C, axis=1)
	# 	_X            = self.X[:,i]  # design matrix excluding non-contrast columns
	# 	_C            = self.C[i]
	#
	# 	# print( _X.shape, _C.shape )
	#
	#
	# 	# PX      = self.X @ np.linalg.pinv(self.X)      # X projector
	# 	# H       = np.linalg.pinv( self.X.T ) @ self.C  # text between eqns. 9.17 & 9.18 (Friston 2007, p.136)
	# 	# PH      = H @ np.linalg.inv(H.T @ H) @ H.T     # H projector
	# 	#
	# 	# X0 = PX - PH
	# 	#
	# 	# print( rank(X0) )
	#
	# 	# trRV,trRVRV   = traceRV(self.V, _X)
	# 	# trMV,trMVMV   = traceMV(self.V, _X, _C)
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

	
	def fit(self):
		Xi     = np.linalg.pinv( self.X )
		b      = Xi @ self.y
		self.e = self.y - self.X @ b   # residuals






class OneWayANOVAModel(GLM):

	def _build_contrast_matrix(self):
		n        = self.X.shape[1]
		C        = np.zeros( (n-1, n) )
		for i in range(n-1):
			C[i,i]   = 1
			C[i,i+1] = -1
		self.C   = C.T

	def _build_design_matrix(self):
		u        = self.uA
		# JJ       = [(self.A==uu).sum()  for uu in u]
		n        = u.size
		X        = np.zeros( (self.J,n) )
		for i,uu in enumerate(u):
			X[:,i]  = self.A==uu
		# X[:,n]   = 1
		self.X   = X
		


	# def _build_contrast_matrix(self):
	# 	n        = self.X.shape[1]
	# 	C        = np.zeros( (n-2, n) )
	# 	for i in range(n-2):
	# 		C[i,i]   = 1
	# 		C[i,i+1] = -1
	# 	self.C   = C.T
	#
	# def _build_design_matrix(self):
	# 	u        = self.uA
	# 	# JJ       = [(self.A==uu).sum()  for uu in u]
	# 	n        = u.size
	# 	X        = np.zeros( (self.J,n+1) )
	# 	for i,uu in enumerate(u):
	# 		X[:,i]  = self.A==uu
	# 	X[:,n]   = 1
	# 	self.X   = X

	def build_variance_model(self, equal_var=False):
		if equal_var:
			Q  = [np.eye(self.J)]
		else:
			Q  = [np.asarray(np.diag( self.A==uu ), dtype=float)  for uu in self.uA]
		self.QQ  = Q
		
		
	
	def calculate_effective_df(self):
		# i             = np.any(self.C, axis=1)
		# _X            = self.X[:,i]  # design matrix excluding non-contrast columns
		# _C            = self.C[i]
		trRV,trRVRV   = traceRV(self.V, self.X[:,:])
		trMV,trMVMV   = traceMV(self.V, self.X, self.C)
		df0           = max(trMV**2 / trMVMV, 1.0)
		df1           = trRV**2 / trRVRV
		v0,v1         = trMV, trRV
		self.df       = df0, df1
		self._v       = v0, v1
	
	
	
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
		V,h           = _reml_old(YY, self.X, self.QQ)
		V            *= (n / np.trace(V))
		self.V        = V
		self.h        = h
		
	# def estimate_variance(self, equal_var=False):
	# 	i             = np.any(self.C, axis=1)
	# 	_X            = self.X[:,i]  # design matrix excluding non-contrast columns
	# 	_C            = self.C[i]
	# 	if equal_var:
	# 		Q  = [np.eye(self.J)]
	# 	else:
	# 		Q  = [np.asarray(np.diag( self.A==uu ), dtype=float)  for uu in self.uA]
	# 	n,s           = self.y.shape
	# 	trRV          = n - rank(_X)
	# 	ss            = (self.e**2).sum(axis=0)
	# 	q             = np.diag(  np.sqrt( trRV / ss )  ).T
	# 	Ym            = self.y @ q
	# 	YY            = Ym @ Ym.T / s
	# 	V,h           = reml(YY, _X, Q)
	# 	V            *= (n / np.trace(V))
	# 	self.V        = V

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
	





class OneWayRMANOVAModel(OneWayANOVAModel):
	def __init__(self, y, A, SUBJ, Q=None):
		self.S  = SUBJ
		self.uS = np.unique( self.S )
		super().__init__(y, A, Q=Q)
		#
		# self.ndim = y.ndim
		# self.y    = y if (self.ndim==2) else np.array( [y] ).T
		# self.A    = np.asarray(A, dtype=float).flatten()
	
	
	def _build_contrast_matrix(self):
		# n        = self.X.shape[1]
		# C        = np.zeros( (n-2, n) )
		# for i in range(n-2):
		# 	C[i,i]   = 1
		# 	C[i,i+1] = -1
		# self.C   = C.T
		
		n  = self.uA.size
		nz = self.X.shape[1] - n
		z  = np.zeros(  (n-1,  nz)  )
		
		C        = np.zeros( (n-1, n) )
		for i in range(n-1):
			C[i,i]   = 1
			C[i,i+1] = -1
		C        = np.hstack([C, z])
		self.C   = C.T
		
		
		
		# z  = np.zeros(  (n-1,  10)  )
		# c  = np.array([ [1,-1,0,0], [0,1,-1,0], [0,0,1,-1] ])
		# c  = np.hstack([c, z])
		# self.C = c.T
		
		
		# z  = np.zeros(  (4,  10)  )
		# c  = -1/3 * np.ones((4,4))
		# np.fill_diagonal(c, 1)
		# c  = np.hstack([c, z])
		# self.C = c.T

	def _build_design_matrix(self):
		# XA   =
		#
		# design = designs.ANOVA1rm(A, SUBJ)
		# design.term_labels = ['A', 'S']
		# XCONST         = design._get_column_const()
		# XA             = design.A.get_design_main()
		# XS             = design.S.get_design_main()
		# # XSA            = self.A.get_design_interaction(self.S)
		# ### specify builder and add design matrix columns:
		# builder        = DesignBuilder(labels=design.term_labels)
		# # builder.add_main_columns('Intercept', XCONST)
		# builder.add_main_columns('A', XA)
		# builder.add_main_columns('S', XS)
		# # builder.add_main_columns('SA', XSA)
		# ### assemble design matrix and contrasts:
		# self.X         = builder.get_design_matrix()

		u        = self.uA
		# JJ       = [(self.A==uu).sum()  for uu in u]
		n        = u.size
		X        = np.zeros( (self.J,n) )
		for i,uu in enumerate(u):
			X[:,i]  = self.A==uu
		XA       = X
		
		XS = []
		for u in self.uS:
			x        = np.zeros(self.J)
			x[self.S==u] =  1
			XS.append(x)
		XS = np.vstack(XS).T
		
		# us       = self.uS
		# ns       = us.size
		# XS       = np.zeros( (self.J,ns) )
		# for i,(a,s) in enumerate( zip(self.A,self.S) ):
		# 	XS[i,s]   = XA
		
		self.X   = np.hstack( [XA, XS] )
		

	def calculate_effective_df(self):
		i             = np.any(self.C, axis=1)
		_X            = self.X[:,i]  # design matrix excluding non-contrast columns
		_C            = self.C[i]
		
		# print( _X.shape, _C.shape )
		
		
		# PX      = self.X @ np.linalg.pinv(self.X)      # X projector
		# H       = np.linalg.pinv( self.X.T ) @ self.C  # text between eqns. 9.17 & 9.18 (Friston 2007, p.136)
		# PH      = H @ np.linalg.inv(H.T @ H) @ H.T     # H projector
		#
		# X0 = PX - PH
		#
		# print( rank(X0) )
		
		# trRV,trRVRV   = traceRV(self.V, _X)
		# trMV,trMVMV   = traceMV(self.V, _X, _C)

		# trRV,trRVRV   = traceRV(self.V, self.X)
		trRV,trRVRV   = traceRV(self.V, self.X[:,:-1])
		trMV,trMVMV   = traceMV(self.V, self.X, self.C)
		# trMV,trMVMV   = traceMV(self.V, _X, _C)

		# print('anova1_oop:',  trRV, trRVRV, trMV, trMVMV)
		df0           = max(trMV**2 / trMVMV, 1.0)
		df1           = trRV**2 / trRVRV
		v0,v1         = trMV, trRV
		
		# print(v1)
		
		# df1,v1 = 27.0, 27.0
		
		
		self.df       = df0, df1
		self._v       = v0, v1
		

	def build_variance_model(self, equal_var=False):
		if equal_var:
			Q  = [np.eye(self.J)]
		else:
			Q  = [np.asarray(np.diag( self.A==uu ), dtype=float)  for uu in self.uA]
			
			# q  = np.ones( (self.J, self.J) )
			# q  = q - np.eye(self.J)
			# Q.append(q)
			

			n  = (self.A == self.uA[0]).sum()
			for i,a0 in enumerate(self.uA):
				for a1 in self.uA[i+1:]:
					q   = np.zeros( (self.J, self.J) )
					i0  = np.argwhere(self.A==a0).flatten()  # rows
					i1  = np.argwhere(self.A==a1).flatten()  # columns
					for ii0,ii1 in zip(i0,i1):
						q[ii0,ii1] = 1
					Q.append( q + q.T )
			
			# Q.pop(3)
			
		self.QQ  = Q
	
	
	def estimate_variance(self):
		i             = np.any(self.C, axis=1)
		_X            = self.X[:,i]  # design matrix excluding non-contrast columns
		_C            = self.C[i]
		# if equal_var:
		# 	Q  = [np.eye(self.J)]
		# else:
		# 	Q  = [np.asarray(np.diag( self.A==uu ), dtype=float)  for uu in self.uA]
		# 	n  = (A == self.uA[0]).sum()
		# 	# zz = np.zeros((n,n))
		# 	# qq = np.eye(n)
		# 	for ii,uu in enumerate(self.uA[:-1]):
		# 		nn = n*(ii-1)
		# 		q  = np.diag(A==uu, nn)[:nn,:nn]
		# 		Q.append( q )
			
			
			# for a0 in self.uA[:-1]:
			# 	for a1 in self.uA[1:]:
			# 		q  = np.zeros( (self.J, self.J) )
			# 		nq = A==a0
			# 		qq = np.eye( )
			# QS = [np.asarray(np.diag( self.S==ss ), dtype=float)  for ss in self.uS]
			# Q  = Q + QS
			
			
			# from math import exp
			# J = self.J
			# q = np.zeros( (J,J) )
			# for i in range(J):
			# 	for ii in range(J):
			# 		if i!=ii:
			# 			q[i,ii] = exp( -abs(i-ii) )
			# Q.append( q )

			
		n,s           = self.y.shape
		# trRV          = n - rank(_X)
		trRV          = n - rank(self.X)
		ss            = (self.e**2).sum(axis=0)
		q             = np.diag(  np.sqrt( trRV / ss )  ).T
		Ym            = self.y @ q
		# Ym            = self.e @ q
		YY            = Ym @ Ym.T / s
		# V,h           = 1(YY, _X, Q)
		# print([trRV, ss])
		# print( np.around(YY,4)[:,:5] )
		# print(Ym)
		
		V,h           = reml(YY, self.X, self.QQ)
		V            *= (n / np.trace(V))
		self.h        = h
		self.V        = V
		# self.QQ       = Q


	def greenhouse_geisser(self):
		
		# this works for 0D data:
		if self.ndim==1:
			y      = self.y.ravel()
			ytable = np.vstack(  [y[self.A==u]  for u in self.uA]  )
			k,n    = ytable.shape
			S      = np.cov(ytable, ddof=1)
		else:
			S       = np.array([h*q  for h,q in zip(self.h, self.QQ)]).sum(axis=0)  # Eqn.10.14  (Friston, 2007)
			k       = self.uA.size
			n       = self.J / k
			
		
		
		# # my BAD 1D attempt:
		# k      = self.uA.size
		# n      = self.J / k
		# S      = self.V


		# # new 1D attempt:
		# S       = np.array([h*q  for h,q in zip(self.h, self.QQ)]).sum(axis=0)  # Eqn.10.14  (Friston, 2007)
		# k       = self.uA.size
		# n       = self.J / k
		
		# eps    = _greenhouse_geisser( S, k )
		
		
		w,v     = np.linalg.eig( S )
		ind     = w.argsort()[::-1]
		v       = v[:,ind]
		M       = v[:,:k-1]
		Sz      = M.T @ S @ M
		w,_     = np.linalg.eig( Sz )
		e       = w.sum()**2 / (   (k-1) * (w**2).sum()   )   # Eqn.13.37, Friston et al. (2007), p.176
		
		self.df  = (k-1)*e, (n-1)*(k-1)*e    # Eqn.13.36, Friston et al. (2007), p.176
		
		




# def anova1_oop(y, A, equal_var=False):
# 	model  = OneWayANOVAModel(y, A)
# 	model.fit()
# 	model.estimate_variance( equal_var=equal_var )
# 	model.calculate_effective_df()
# 	model.calculate_f_stat()
# 	f,df   = model.f, model.df
#
# 	if isinstance(f, float):
# 		p  = scipy.stats.f.sf(f, df[0], df[1])
# 	else:
# 		fwhm    = rft1d.geom.estimate_fwhm( model.e )
# 		p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
#
# 	return f, df, p, model
#
#
#
# def anova1rm_oop(y, A, SUBJ, equal_var=False, gg=True):
# 	model  = OneWayRMANOVAModel(y, A, SUBJ)
# 	model.build_variance_model( equal_var=equal_var )
# 	model.fit()
# 	model.estimate_variance(  )
# 	model.calculate_effective_df()
# 	model.calculate_f_stat()
# 	if gg:
# 		model.greenhouse_geisser()
#
# 	f,df,V   = model.f, model.df, model.V
#
# 	if isinstance(f, float):
# 		p  = scipy.stats.f.sf(f, df[0], df[1])
# 	else:
# 		fwhm    = rft1d.geom.estimate_fwhm( model.e )
# 		p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
# 	return f, df, p, model
#
# anova1rm = anova1rm_oop


# def anova1(y, A, equal_var=False):
# 	ndim    = y.ndim
# 	J       = y.shape[0]
# 	u       = np.unique(A)
# 	JJ      = [(A==uu).sum()  for uu in u]
# 	nf      = u.size
# 	# # design matrix
# 	# X       = np.zeros( (J,nf) )
# 	# X[:,0]  = 1
# 	# for i,uu in enumerate(u[1:]):
# 	# 	X[:,i+1]  = A==uu
# 	# design matrix
# 	X       = np.zeros( (J,nf) )
# 	for i,uu in enumerate(u):
# 		X[:,i]  = A==uu
# 	# reduced design matrix
# 	Xr      = np.ones( (J,1) )
# 	Y       = y if (ndim==2) else np.array( [y] ).T
# 	# contrast matrix:
# 	c             = np.zeros( (nf-1, nf) )
# 	for i in range(nf-1):
# 		c[i,i]   = 1
# 		c[i,i+1] = -1
# 	c = c.T
#
# 	# residuals:
# 	Xi   = np.linalg.pinv(X)
# 	b    = Xi @ Y
# 	eij  = Y - X @ b
# 	# covariance model:
# 	if equal_var:
# 		Q  = [np.eye(J)]
# 	else:
# 		Q  = [np.asarray(np.diag( A==uu ), dtype=float)  for uu in u]
# 	# estimate covariance and DF
# 	n,s           = Y.shape
# 	trRV          = n - rank(X)
# 	ResSS         = (np.asarray(eij)**2).sum(axis=0)
# 	q             = np.diag(np.sqrt( trRV / ResSS )).T
# 	Ym            = Y @ q
# 	YY            = Ym @ Ym.T / s
# 	V,h           = reml(YY, X, Q)
# 	V            *= (n / np.trace(V))
# 	### effective degrees of freedom (denominator):
# 	trRV,trRVRV   = traceRV(V, X)
# 	df1           = trRV**2 / trRVRV
# 	### effective degrees of freedom (numerator):
# 	trMV,trMVMV   = traceMV(V, X, c)
# 	# print('anova1:  f=%0.3f, df=(%0.3f,f=%0.3f), p=%0.5f', )
# 	# print('anova1:',  trRV, trRVRV, trMV, trMVMV)
# 	df0           =  trMV**2 / trMVMV
# 	df0           = max(df0,1.0)
# 	v0,v1         = trMV, trRV
#
# 	# F stat
# 	I       = np.eye(J)
# 	P       = X @ np.linalg.inv(X.T @ X) @ X.T
# 	Pr      = Xr @ np.linalg.inv(Xr.T @ Xr) @ Xr.T
# 	YIPY    = Y.T @ ( I - P ) @ Y
# 	YIPrY   = Y.T @ ( I - Pr ) @ Y
# 	f       = ((YIPrY - YIPY) / v0) / ( YIPY / v1 )
# 	f       = float(f) if (ndim==1) else np.diag(f)
#
# 	if ndim==1:
# 		p       = scipy.stats.f.sf(f, df0, df1)
# 	else:
# 		Q       = y.shape[1]
# 		efwhm   = rft1d.geom.estimate_fwhm( eij )
# 		p       = rft1d.f.sf(f.max(), (df0, df1), Q, efwhm)
#
# 	return f, (df0,df1), V, p
#
# 	# return f, None, None, None
#

