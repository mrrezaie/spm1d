
import pytest


def test_rft1d_location():
	import rft1d
	assert 'spm1d' not in rft1d.__file__

