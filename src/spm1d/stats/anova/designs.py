
import numpy as np
from . factors import Factor

class _Design(object):
	def __init__(self):
		self.X             = None   # design matrix
		self.contrasts     = None   # contrast objects
		self.factors       = None   # list of factor objects
		
	def _init_factors(self, *AA):
		self.factors = [Factor(A, name=chr(65+i))   for i,A in enumerate(AA)]
		# s0,s1        = factor_names
		# ss0,ss1      = factor_names_s
		# self.factors = [ Factor(A, name=s0, name_s=ss0), Factor(B, name=s1, name_s=ss1) ]

	@property
	def J(self):
		return self.X.shape[0]
	@property
	def nfactors(self):
		return len( self.factors )
		
	def _assemble(self):
		self.X         = self._build_design_matrix()
		self.contrasts = self._build_contrasts()


	def set_factor_names(self, names, names_short=None):
		# self.set_factor_names(names, names_short)
		if names_short is None:
			names_short = [None] * self.nfactors
		for factor,s,ss in zip(self.factors, names, names_short):
			factor.set_name( s, ss )





class ANOVA1(_Design):
	def __init__(self, A):
		self.factors      = [ Factor(A, name='A') ]
		self._assemble()

	def _build_contrasts(self):
		n        = self.factors[0].nlevels
		C        = np.zeros( (n-1, n) )
		for i in range(n-1):
			C[i,i]   = 1
			C[i,i+1] = -1
		return [C.T]

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

	# def _build_contrasts(self):
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
		
	
	
	def _build_contrasts(self):
		n        = self.factors[0].nlevels
		C        = np.zeros( (n-1, n) )
		for i in range(n-1):
			C[i,i]   = 1
			C[i,i+1] = -1
			
		nz       = self.factors[1].nlevels
		Cz       = np.zeros(  (n-1,  nz)  )
		C        = np.hstack([C, Cz])
		return [C.T]
		

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
		
		
		
	# def get_variance_model(self, equal_var=False):
	# 	if equal_var:
	# 		Q   = [np.eye(self.J)]
	# 	else:
	# 		A,u = self.factors[0].A, self.factors[0].u
	# 		Q   = [np.asarray(np.diag( A==uu ), dtype=float)  for uu in u]
	# 	return Q


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




class ANOVA2(_Design):
	def __init__(self, A, B):
		self._init_factors( A, B )
		self._assemble()


	# def _init_factors(self, A, B):
	# 	s0,s1        = factor_names
	# 	ss0,ss1      = factor_names_s
	# 	self.factors = [ Factor(A, name=s0, name_s=ss0), Factor(B, name=s1, name_s=ss1) ]

	def _build_contrasts(self):
		from . contrasts import Contrast
		
		fA,fB = self.factors
		n     = self.X.shape[1]
		nA    = fA.n - 1
		nB    = fB.n - 1
		nAB   = nA * nB
		
		CA = []
		for i in range(nA):
			c   = np.zeros(n)
			c[i] = 1
			CA.append(c)
		# CA = Contrast( np.asarray(CA).T, name=f'Main {fA.name}', name_s=fA.name_s )
		CA = Contrast( np.asarray(CA).T, factors=[fA] )


		CB = []
		for i in range(nB):
			c   = np.zeros(n)
			c[nA+i] = 1
			CB.append(c)
		# CB = Contrast( np.asarray(CB).T, name=f'Main {fB.name}', name_s=fB.name_s )
		CB = Contrast( np.asarray(CB).T, factors=[fB] )
		
		

		CAB  = []
		for i in range(nAB):
			c   = np.zeros(n)
			c[nA+nB+i] = 1
			CAB.append(c)
		# CAB = Contrast( np.asarray(CAB).T, name=f'Interaction {fA.name} x {fB.name}', name_s=f'{fA.name_s}x{fB.name_s}' )
		CAB = Contrast( np.asarray(CAB).T, factors=[fA,fB] )
	
		C    = [CA, CB, CAB]
		
		return C
		

	def _build_design_matrix(self):
		fA,fB     = self.factors
		XA        = fA.get_design_mway_main()
		XB        = fB.get_design_mway_main()
		XAB       = np.asarray(  [np.kron( XA[i], XB[i] )   for i in range(XA.shape[0])] )
		X0        = fA.get_design_intercept()
		X         = np.hstack( [XA, XB, XAB, X0] )
		return X
		
	def get_variance_model(self, equal_var=False):
		pass
		# if equal_var:
		# 	Q   = [np.eye(self.J)]
		# else:
		# 	A,u = self.factors[0].A, self.factors[0].u
		# 	Q   = [np.asarray(np.diag( A==uu ), dtype=float)  for uu in u]
		# return Q