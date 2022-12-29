
from . import *
# from . import uv0d, mv0d
# from . import uv1d, mv1d


def iter_0d_ttest():
	yield RSWeightReduction()
	yield ColumbiaSalmonella()

	




# dsets.append(   [spm1d.data.,    spm1d.stats.ttest]   )
# dsets.append(   [spm1d.data.uv0d.t1.ColumbiaSalmonella,   spm1d.stats.ttest]   )
# dsets.append(   [spm1d.data.uv0d.tpaired.RSWeightClinic,  spm1d.stats.ttest_paired]   )
# dsets.append(   [spm1d.data.uv0d.tpaired.ColumbiaMileage, spm1d.stats.ttest_paired]   )

