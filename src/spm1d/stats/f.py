'''
ANOVA
'''

# Copyright (C) 2023  Todd Pataky



import numpy as np
import scipy.stats
import rft1d
from . _cov import reml, traceRV, traceMV
from . _la import rank




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


