

from . import uv0d, mv0d
from . import uv1d, mv1d



RSWeightReduction  = uv0d.t1.RSWeightReduction
ColumbiaSalmonella = uv0d.t1.ColumbiaSalmonella


# dsets.append(   [spm1d.data.,    spm1d.stats.ttest]   )
# dsets.append(   [spm1d.data.uv0d.t1.ColumbiaSalmonella,   spm1d.stats.ttest]   )
# dsets.append(   [spm1d.data.uv0d.tpaired.RSWeightClinic,  spm1d.stats.ttest_paired]   )
# dsets.append(   [spm1d.data.uv0d.tpaired.ColumbiaMileage, spm1d.stats.ttest_paired]   )




__all__ = [
	'ColumbiaSalmonella',
	'RSWeightReduction',
]
