
import numpy as np


def _round(x, n):
	if isinstance(x, (int,float,np.int64)):
		xr = round(x, n)
	elif isinstance(x, tuple):
		xr = tuple(round(xx, n) for xx in x)
	elif isinstance(x, list):
		xr = [round(xx, n) for xx in x]
	elif isinstance(x, np.ndarray):
		xr = np.around(x, n)
	return xr

class Cluster04(object):
	def __init__(self, cluster):
		self.centroid  = _round(cluster.centroid, 5)
		self.endpoints = _round(cluster.endpoints, 5)
		self.extent    = _round(cluster.extent, 5)
	def __repr__(self):
		return str(self.__dict__)

class SPM04(object):
	def __init__(self, spm):
		self.STAT      = spm.STAT
		self.z         = _round(spm.z, 5)
		self.df        = spm.df
		self.fwhm      = _round(spm.fwhm, 5)
		self.resels    = _round(spm.resels, 5)
		self.zstar     = _round(spm.zstar, 5)
		self.clusters  = [Cluster04(c) for c in spm.clusters]
	
	def __repr__(self):
		s      = 'SPM04 (spm1d-v0.4 proxy)\n'
		for k,v in self.__dict__.items():
			s += f'   {k:<10} : {v}\n'
		return s

class SPM04TestCase(object):
	def __init__(self, dataset, testname, infkwargs):
		self.name      = dataset.__class__.__name__
		self.args      = tuple(_round(x, 5) for x in dataset.get_data())
		self.testname  = testname
		self.infkwargs = infkwargs
		self.spmi      = SPM04( self.get_spm() )
		
	def __repr__(self):
		s      = 'SPM04TestCase\n'
		for k,v in self.__dict__.items():
			s += f'   {k:<10} : {v}\n'
		return s

	def get_spm(self):
		fn    = eval( f'spm1d.stats.{self.testname}' )
		# args  = self.dataset.get_data()
		spmi  = fn(*self.args).inference(alpha, **self.infkwargs)
		return spmi

	def save(self, dir1, check_load=True):
		fpath  = os.path.join( dir1   , f'{self.name}.pkl'   )
		with open(fpath, 'wb') as f:
			pickle.dump( self, f, protocol=pickle.HIGHEST_PROTOCOL )
		if check_load:
			with open(fpath, 'rb') as f:
				b = pickle.load(f)
			print(self)
			print(b)

