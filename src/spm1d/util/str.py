

'''
Utility functions (string)
'''



def df2str(v):
	return str(v) if not v%1 else f'{v:.3f}'


def dflist2str(v):
	s0,s1 = df2str(v[0]), df2str(v[1])
	return f'({s0}, {s1})'


def p2string(p):
	return '<0.001' if p<0.0005 else f'{p:.3f}'


def plist2string(pList):
	s      = ''
	if len(pList)>0:
		for p in pList:
			s += p2string(p)
			s += ', '
		s  = s[:-2]
	return s


def plist2stringlist(pList):
	s  = plist2string(pList).split(', ')
	for i,ss in enumerate(s):
		if ss.startswith('<'):
			s[i]  = 'p' + ss
		else:
			s[i]  = 'p=' + ss
	return s