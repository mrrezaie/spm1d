
'''
Utility functions (probability)
'''

# Copyright (C) 2022  Todd Pataky





def p_corrected_bonf(p, n):
	'''
	Bonferroni-corrected *p* value.
	
	.. warning:: This correction assumes independence amongst multiple tests.
	
	:Parameters:
	
	- *p* --- probability value computed from one of multiple tests
	- *n* --- number of tests
	
	:Returns:
	
	- Bonferroni-corrected *p* value.
	
	:Example:
	
	>>> p = spm1d.util.p_corrected_bonf(0.03, 8)    # yields p = 0.216
	'''
	if p<=0:
		return 0
	elif p>=1:
		return 1
	else:
		pBonf  = 1 - (1.0-p)**n
		pBonf  = max(0, min(1, pBonf))
		return pBonf



def p_critical_bonf(alpha, n):
	'''
	Bonferroni-corrected critical Type I error rate.
	
	.. warning:: This crticial threshold assumes independence amongst multiple tests.
	
	:Parameters:
	
	- *alpha* --- original Type I error rate (usually 0.05)
	- *n* --- number of tests
	
	:Returns:
	
	- Bonferroni-corrected critical *p* value; retains *alpha* across all tests.
	
	:Example:
	
	>>> p = spm1d.util.p_critical_bonf(0.05, 20)    # yields p = 0.00256
	'''
	if alpha<=0:
		return 0
	elif alpha>=1:
		return 1
	else:
		return 1 - (1.0-alpha)**(1.0/n)
