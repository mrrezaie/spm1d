
import numpy as np
import rft1d
from . cls import Cluster




def _cluster_geom(z, u, interp=True, circular=False, csign=+1, ClusterClass=Cluster):
	Q        = z.size
	Z        = z
	X        = np.arange(Q)
	if np.ma.is_masked(Z):
		i    = Z.mask
		Z    = np.array(Z)
		B    = (csign*Z) >= u
		B[i] = False
		Z[i] = np.nan
	else:
		B    = (csign*Z) >= u
	Z        = csign*Z
	L,n      = rft1d.geom.bwlabel(B)
	clusters = []
	for i in range(n):
		b    = L==(i+1)
		x,z  = X[b].tolist(), Z[b].tolist()
		# interpolate to threshold u using similar triangles method
		# (interpolate for plotting whether or not "interp" is true)
		if (x[0]>0) and not np.isnan( Z[x[0]-1] ):  #first cluster point not continuum edge && previous point not outside ROI
			z0,z1  = Z[x[0]-1], Z[x[0]]
			dx     = (z1-u) / (z1-z0)
			x      = [x[0]-dx] + x
			z      = [u] +z
		if (x[-1]<Q-1) and not np.isnan( Z[x[-1]+1] ):  #last cluster point not continuum edge && next point not outside ROI
			z0,z1  = Z[x[-1]], Z[x[-1]+1]
			dx     = (z0-u) / (z0-z1)
			x     += [x[-1]+dx]
			z     += [u]
		# create cluster:
		x,z  = np.array(x), csign*np.array(z)
		clusters.append(  ClusterClass(x, z, csign*u, interp) )
	#merge clusters if necessary (circular fields only)
	if circular and ( len(clusters)>0 ):
		xy         = np.array([c.endpoints  for c in clusters])
		i0,i1      = xy[:,0]==0, xy[:,1]==Q-1
		ind0,ind1  = np.argwhere(i0), np.argwhere(i1)
		if (len(ind0)>0) and (len(ind1)>0):
			ind0,ind1 = ind0[0][0], ind1[0][0]
			if (ind0!=ind1) and (clusters[ind0].csign == clusters[ind1].csign):
				clusters[ind0].merge( clusters[ind1] )
				clusters.pop( ind1 )
	return clusters
	
	
def assemble(z, zc, interp=True, circular=False, check_neg=False):
	clusters      = _cluster_geom(z, zc, interp, circular, csign=+1)
	if check_neg:
		clustersn = _cluster_geom(z, zc, interp, circular, csign=-1)
		clusters += clustersn
		if len(clusters) > 1:
			### reorder clusters left-to-right:
			x         = [c.xy[0]  for c in clusters]
			ind       = np.argsort(x).flatten()
			clusters  = np.array(clusters)[ind].tolist()
	return clusters