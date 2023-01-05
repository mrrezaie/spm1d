

'''
Utility functions (string)
'''



def df2str(v):
	return str(v) if not v%1 else f'{v:.3f}'


def dflist2str(v):
	s0,s1 = df2str(v[0]), df2str(v[1])
	return f'({s0}, {s1})'


def p2string(p, allow_none=False):
	if allow_none and (p is None):
		s = 'None'
	else:
		s = '<0.001' if p<0.0005 else f'{p:.3f}'
	return s


def plist2string(plist):
	return ', '.join( [p2string(p) for p in plist] )

def plist2stringlist(plist):
	s  = plist2string(plist).split(', ')
	for i,ss in enumerate(s):
		if ss.startswith('<'):
			s[i]  = 'p' + ss
		else:
			s[i]  = 'p=' + ss
	return s
	
def tuple2str(x, fmt='%.3f'):
	return '(' +  ', '.join( (fmt%xx for xx in x) ) + ')'
