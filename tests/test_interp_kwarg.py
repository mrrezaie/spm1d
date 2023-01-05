
import pytest
import warnings
import spm1d
from spm1d.cfg import SPM1DDeprecationWarning




def test_interp_kwarg():
	dataset = spm1d.data.uv1d.t1.SimulatedPataky2015b()
	Y,mu    = dataset.get_data()
	t       = spm1d.stats.ttest(Y, mu)
	# ensure that a deprecation warning is issued
	spm1d.cfg.enable_warnings( True )
	with pytest.deprecated_call():
		ti  = t.inference(0.05, method='rft', dirn=1, interp=True)
	# # ensure that no warning is issued:
	# spm1d.cfg.enable_warnings( True )
	# with pytest.warns() as w:
	# 	ti  = t.inference(0.05, method='rft', interp=True)
	# 	x   = print( ti.zstar )
	# 	assert len(w) == 1
	# 	assert w[0].category == SPM1DDeprecationWarning
		

