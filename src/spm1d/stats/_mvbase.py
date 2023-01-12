
# Copyright (C) 2023  Todd Pataky

from math import sqrt,log
import numpy as np
from scipy import ndimage

eps        = np.finfo(float).eps   #smallest float, used to avoid divide-by-zero errors






def _get_residuals_onesample(Y):
	N = Y.shape[0]
	m = Y.mean(axis=0)
	R = Y - np.array([m]*N)
	return R

def _get_residuals_onesample_0d(Y):
	N = Y.shape[0]
	m = Y.mean(axis=0)
	R = Y - np.array([m]*N)
	return R

def _get_residuals_twosample(YA, YB):
	RA = _get_residuals_onesample(YA)
	RB = _get_residuals_onesample(YB)
	R  = np.vstack( (RA,RB) )
	return R


def _get_residuals_twosample_0d(YA, YB):
	RA = _get_residuals_onesample_0d(YA)
	RB = _get_residuals_onesample_0d(YB)
	R  = np.hstack( (RA,RB) )
	return R


def _get_residuals_regression(y, x):
	J,Q,I      = y.shape  #nResponses, nNodes, nComponents
	Z          = np.matrix(np.ones(J)).T
	X          = np.hstack([np.matrix(x.T).T, Z])
	Xi         = np.linalg.pinv(X)
	R          = np.zeros(y.shape)
	for i in range(Q):
		for ii in range(I):
			yy     = np.matrix(y[:,i,ii]).T
			b      = Xi*yy
			eij    = yy - X*b
			R[:,i,ii] = np.asarray(eij).flatten()
	return R

def _get_residuals_regression_0d(y, x):
	J,I         = y.shape  #nResponses, nNodes, nComponents
	Z           = np.matrix(np.ones(J)).T
	X           = np.hstack([np.matrix(x.T).T, Z])
	Xi          = np.linalg.pinv(X)
	R           = np.zeros(y.shape)
	for ii in range(I):
		yy      = np.matrix(y[:,ii]).T
		b       = Xi*yy
		eij     = yy - X*b
		R[:,ii] = np.asarray(eij).flatten()
	return R


def _get_residuals_manova1(Y, GROUP):
	u  = np.unique(GROUP)
	R  = []
	for uu in u:
		R.append(   _get_residuals_onesample(Y[GROUP==uu])   )
	return np.vstack(R)

def _get_residuals_manova1_0d(Y, GROUP):
	u  = np.unique(GROUP)
	R  = []
	for uu in u:
		R.append(   _get_residuals_onesample_0d(Y[GROUP==uu])   )
	return np.hstack(R)













