
from copy import deepcopy
from math import inf
from .. _argparse import ArgumentParser


def _force_iterations_deprecation_msg():
	msg  = '\n\n\n'
	msg += 'WARNING!! The keyword argument "force_iterations" will be removed from future versions of spm1d.\n'
	msg += '  - In spm1d-v0.4 "force_iterations=True" was used to confirm a large number of permutations (>1e5) during nonparametric (permutation-based) inference.\n'
	msg += '  - In spm1d-v0.5 just set "nperm" to a large value (e.g. "nperm=1e6") to replicate "force_iterations=True" \n'
	msg += '\n\n\n'
	return msg

def _iterations_deprecation_msg():
	msg  = '\n\n\n'
	msg += 'WARNING!! The keyword argument "iterations" will be removed from future versions of spm1d.\n'
	msg += '  - Use "nperm" in place of "iterations"\n'
	msg += '\n\n\n'
	return msg

def _two_tailed_deprecation_msg():
	msg  = '\n\n\n'
	msg += 'WARNING!! The keyword argument "two_tailed" will be removed from future versions of spm1d.\n'
	msg += '  - Use "dirn=0" in place "two_tailed=True"\n'
	msg += '  - Use "dirn=+1" or "dirn=-1" in place of "two_tailed=False".\n'
	msg += '  - In spm1d-v0.4 and earlier, "two_tailed=False" is equivalent to the new syntax: "dirn=1".\n'
	msg += '  - In spm1d-v0.5 the default is "dirn=0".\n'
	msg += '\n\n\n'
	return msg




class InferenceArgumentParser0D( ArgumentParser ):

	def __init__(self, stat, method):
		super().__init__()
		self.stat     = stat
		self.method   = method
		self.kwargs   = None
		self._init()
	
	def _init(self):
		self.add_arg('alpha', type=float, range=(0,1))
		self.add_kwarg('method', values=['gauss','perm'])
		if self.stat =='T':
			self.add_kwarg(name='dirn', values=[-1,0,1], default=0)
			k = self.add_kwarg(name='two_tailed', values=[True,False,None], default=None)
			k.set_deprecated_warning( _two_tailed_deprecation_msg() )
			self.set_invalid_combinations( ('dirn', 'two_tailed'), [(0,False), (1,True), (-1,True)])
		
		if self.method=='perm':
			k0 = self.add_kwarg(name='iterations', type=int, range=(-1,inf), default=-1, badvalues=list(range(0,10)))
			self.add_kwarg(name='nperm', type=int, range=(-1,inf), default=-1, badvalues=list(range(0,10)))
			k1 = self.add_kwarg(name='force_iterations', type=bool)
			
			k0.set_deprecated_warning( _iterations_deprecation_msg() )
			k1.set_deprecated_warning( _force_iterations_deprecation_msg() )
			self.set_invalid_pairs( ('iterations', 'nperm'))
		
	# def parse(self, alpha, method='gauss', **kwargs):
	# 	super().parse(alpha, method=method, **kwargs)
	# 	self.kwargs = kwargs
	# 	if (self.stat=='T') and ('two_tailed' in kwargs):
	# 		if 'dirn' not in kwargs:
	# 			self.kwargs['dirn'] = 0 if kwargs['two_tailed'] else 1
	# 		self.kwargs.pop( 'two_tailed' )

	def parse(self, alpha, **kwargs):
		super().parse(alpha, method=self.method, **kwargs)
		self.kwargs = kwargs
		if (self.stat=='T') and ('two_tailed' in kwargs):
			if 'dirn' not in kwargs:
				self.kwargs['dirn'] = 0 if kwargs['two_tailed'] else 1
			self.kwargs.pop( 'two_tailed' )
		if (self.method=='perm') and ('iterations' in kwargs):
			self.kwargs['nperm'] = self.kwargs['iterations']
			self.kwargs.pop('iterations')
		if (self.method=='perm') and ('force_iterations' in kwargs):
			self.kwargs.pop('force_iterations')
		



class InferenceArgumentParser1D( InferenceArgumentParser0D ):
	
	def _init(self):
		super()._init()
		self.ek['method'].append_values(  ['rft', 'fmax', 'fdr', 'bonferroni', 'uncorrected'] )
		self.add_kwarg(name='roi', type=np.ndarray)

		if (self.method in ['gauss', 'rft']):
			self.add_kwarg(name='cluster_size', type=(int,float), default=0, range=(0,inf))
			self.add_kwarg(name='interp', type=bool)
			self.add_kwarg(name='circular', type=bool)
			self.add_kwarg(name='withBonf', type=bool)

		elif self.method in ['perm', 'fmax']:
			self.add_kwarg(name='cluster_size', type=(int,float), default=0, range=(0,inf))
			self.add_kwarg(name='interp', type=bool)
			self.add_kwarg(name='circular', type=bool)
			self.add_kwarg(name='cluster_metric', values=self._get_cluster_metrics(), default='MaxClusterExtent')

	def _get_cluster_metrics(self):
		from .. nonparam.metrics import metric_dict
		return list( metric_dict.keys() )



# class InferenceArgumentParser( ArgumentParser ):
#
# 	def __init__(self, stat, dim, method):
# 		super().__init__()
# 		self.stat     = stat
# 		self.dim      = dim
# 		self.method   = method
# 		self._init()
#
# 	def _init(self):
#
# 		a0      = self.add_arg('alpha', type=float, range=(0,1))
# 		k0      = self.add_kwarg('method', values=self._get_methods())
# 		if self.stat =='T':
# 			k1  = self.add_kwarg(name='dirn', values=[-1,0,1], default=0)
# 			k2  = self.add_kwarg(name='two_tailed', values=[True,False,None], default=None)
# 			k2.set_deprecated_warning( _two_tailed_deprecation_msg() )
# 			self.set_invalid_combinations( ('dirn', 'two_tailed'), [(0,False), (1,True), (-1,True)])
#
# 		if self.dim==1:
# 			k   = self.add_kwarg(name='roi', type=np.ndarray)
#
# 		if (self.method in ['gauss', 'rft']) and (self.dim==1):
# 			self.add_kwarg(name='cluster_size', type=(int,float), default=0, range=(0,inf))
# 			self.add_kwarg(name='interp', type=bool)
# 			self.add_kwarg(name='circular', type=bool)
# 			self.add_kwarg(name='withBonf', type=bool)
#
# 		elif self.method=='perm':
# 			self.add_kwarg(name='iterations', type=int, range=(-1,inf), default=-1, badvalues=list(range(0,10)))
# 			self.add_kwarg(name='force_iterations', type=bool)
# 			if self.dim==1:
# 				self.add_kwarg(name='interp', type=bool)
# 				self.add_kwarg(name='circular', type=bool)
# 				self.add_kwarg(name='cluster_metric', values=self._get_cluster_metrics(), default='MaxClusterExtent')
#
#
# 	def _get_cluster_metrics(self):
# 		from .. nonparam.metrics import metric_dict
# 		return list( metric_dict.keys() )
#
# 	def _get_methods(self):
# 		methods = ['gauss','perm']
# 		if self.dim==1:
# 			methods += ['rft', 'fmax', 'fdr', 'bonferroni', 'uncorrected']
# 		return methods
#
#
#
# 	def parse(self, alpha, method='gauss', **kwargs):
# 		super().parse(alpha, method=method, **kwargs)
