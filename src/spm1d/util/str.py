

'''
Utility functions (string)
'''


def array2shortstr(a):
	return f'{a.shape} array'


def arraytuple2str(aa):
	return '(  ' + ',  '.join(  [array2shortstr(a) for a in aa]  ) + '  )'


def df2str(v):
	return str(v) if not v%1 else f'{v:.3f}'


def dflist2str(v):
	s0,s1 = df2str(v[0]), df2str(v[1])
	return f'({s0}, {s1})'


def largeint2str(x, mx=1e9):
	return str(x) if (x < mx) else  f'> {mx:.3E}'
	


def float2string(x, allow_none=False, fmt='%.3f'):
	return 'None' if (allow_none and (x is None)) else fmt%x


def p2string(p, allow_none=False, fmt='%.3f'):
	if allow_none and (p is None):
		s = 'None'
	else:
		s   = '<0.001' if p<0.0005 else fmt%p
	return s


def plist2string(plist, allow_none=False, fmt='%.3f'):
	return '[' + ', '.join( [p2string(p, allow_none=allow_none, fmt=fmt) for p in plist] ) + ']'


def plist2stringlist(plist):
	s  = plist2string(plist).split(', ')
	for i,ss in enumerate(s):
		if ss.startswith('<'):
			s[i]  = 'p' + ss
		else:
			s[i]  = 'p=' + ss
	return s


def resels2str(resels):
	return '(%d, %.5f)'%tuple(resels)
	

def tuple2str(x, fmt='%.3f'):
	return '(' +  ', '.join( (fmt%xx for xx in x) ) + ')'
