
'''
Non-parametric permutation hypothesis tests for 1D data (Statistical non-Parametric Mapping)

This sub-package is no longer used. Instead access permutation using method="perm" like this:

	>>> t  = spm1d.stats.ttest2(yA, yB)
	>>> ti = t.inference(0.05, method="perm", nperm=1000)
'''

# Copyright (C) 2023  Todd Pataky




def _dmy(*args, **kwargs):
	msg  = 'Nonparametric permutation analysis is no longer available through "spm1d.stats.nonparam". '
	msg += f'Instead access permutation inference using "method={chr(39)}perm{chr(39)}" when calling "inference".\n'
	msg += 'For example:\n'
	msg +=  '      t  = spm1d.stats.ttest2(yA, yB)\n'
	msg +=  '      ti = t.inference(0.05, method="perm", nperm=1000)\n'
	raise NotImplementedError(msg)


anova1            = _dmy
anova1rm          = _dmy
anova2            = _dmy
anova2nested      = _dmy
anova2onerm       = _dmy
anova2rm          = _dmy
anova3            = _dmy
anova3nested      = _dmy
anova3onerm       = _dmy
anova3tworm       = _dmy
anova3rm          = _dmy
regress           = _dmy
ttest             = _dmy
ttest_paired      = _dmy
ttest2            = _dmy
cca               = _dmy
hotellings        = _dmy
hotellings_paired = _dmy
hotellings2       = _dmy
manova1           = _dmy
ci_onesample      = _dmy
ci_pairedsample   = _dmy
ci_twosample      = _dmy
