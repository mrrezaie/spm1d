
import numpy as np
from . factors import Factor

class _Design(object):
	def __init__(self):
		self.X             = None   # design matrix
		self.C             = None   # contrast matrix
		self.factors       = None   # list of factor objects
		
	@property
	def J(self):
		return self.X.shape[0]
		
		


class ANOVA1(_Design):
	def __init__(self, A):
		self.factors      = [ Factor(A, name='A') ]
		self._assemble()

	def _assemble(self):
		self.X  = self._build_design_matrix()
		self.C  = self._build_contrast_matrix()

	def _build_contrast_matrix(self):
		n        = self.factors[0].nlevels
		C        = np.zeros( (n-1, n) )
		for i in range(n-1):
			C[i,i]   = 1
			C[i,i+1] = -1
		return C.T

	def _build_design_matrix(self):
		return self.factors[0].get_design_main()
		
	def get_variance_model(self, equal_var=False):
		if equal_var:
			Q   = [np.eye(self.J)]
		else:
			A,u = self.factors[0].A, self.factors[0].u
			Q   = [np.asarray(np.diag( A==uu ), dtype=float)  for uu in u]
		return Q
	
	
	
class ANOVA1RM(_Design):
	def __init__(self, A, SUBJ):
		self.factors      = [ Factor(A, name='A'), Factor(SUBJ, name='SUBJ') ]
		self._assemble()

	def _assemble(self):
		self.X  = self._build_design_matrix()
		self.C  = self._build_contrast_matrix()

	# def _build_contrast_matrix(self):
	# 	n        = self.factors[0].nlevels
	# 	C        = np.zeros( (n-1, n) )
	# 	for i in range(n-1):
	# 		C[i,i]   = 1
	# 		C[i,i+1] = -1
	# 	return C.T
	#
	# def _build_design_matrix(self):
	# 	return self.factors[0].get_design_main()
		
	# def get_variance_model(self, equal_var=False):
	# 	if equal_var:
	# 		Q   = [np.eye(self.J)]
	# 	else:
	# 		A,u = self.factors[0].A, self.factors[0].u
	# 		Q   = [np.asarray(np.diag( A==uu ), dtype=float)  for uu in u]
	# 	return Q
		
	
	
	def _build_contrast_matrix(self):
		n        = self.factors[0].nlevels
		C        = np.zeros( (n-1, n) )
		for i in range(n-1):
			C[i,i]   = 1
			C[i,i+1] = -1
			
		nz       = self.factors[1].nlevels
		Cz       = np.zeros(  (n-1,  nz)  )
		C        = np.hstack([C, Cz])
		return C.T
		

		# n  = self.uA.size
		# nz = self.X.shape[1] - n
		# z  = np.zeros(  (n-1,  nz)  )
		# C  = np.zeros( (n-1, n) )
		# for i in range(n-1):
		# 	C[i,i]   = 1
		# 	C[i,i+1] = -1
		# C        = np.hstack([C, z])
		# self.C   = C.T

	def _build_design_matrix(self):
		XA       = self.factors[0].get_design_main()
		XS       = self.factors[1].get_design_main()
		# XS       =
		#
		#
		# u        = self.uA
		# n        = u.size
		# X        = np.zeros( (self.J,n) )
		# for i,uu in enumerate(u):
		# 	X[:,i]  = self.A==uu
		# XA       = X
		
		# XS = []
		# for u in self.uS:
		# 	x        = np.zeros(self.J)
		# 	x[self.S==u] =  1
		# 	XS.append(x)
		# XS = np.vstack(XS).T
		
		return np.hstack( [XA, XS] )
		
		
		
	def get_variance_model(self, equal_var=False):
		if equal_var:
			Q   = [np.eye(self.J)]
		else:
			A,u = self.factors[0].A, self.factors[0].u
			Q   = [np.asarray(np.diag( A==uu ), dtype=float)  for uu in u]
		return Q


	def get_variance_model(self, equal_var=False):
		if equal_var:
			Q  = [np.eye(self.J)]
		else:
			A,u = self.factors[0].A, self.factors[0].u
			Q   = [np.asarray(np.diag( A==uu ), dtype=float)  for uu in u]
			
			n   = (A == u[0]).sum()
			for i,a0 in enumerate(u):
				for a1 in u[i+1:]:
					q   = np.zeros( (self.J, self.J) )
					i0  = np.argwhere(A==a0).flatten()  # rows
					i1  = np.argwhere(A==a1).flatten()  # columns
					for ii0,ii1 in zip(i0,i1):
						q[ii0,ii1] = 1
					Q.append( q + q.T )
		return Q
	