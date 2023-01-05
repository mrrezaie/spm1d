
import pytest
import spm1d



def test_cluster_info():
	dataset = spm1d.data.uv1d.t1.SimulatedPataky2015b()
	Y,mu    = dataset.get_data()
	t       = spm1d.stats.ttest(Y, mu)
	ti      = t.inference(0.05, method='rft', dirn=1, circular=False)
	assert hasattr(ti, 'p_max')


