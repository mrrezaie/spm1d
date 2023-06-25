
import scipy.stats
import rft1d
from . models import GeneralLinearModel
# from . _glm import OneWayANOVAModel, OneWayRMANOVAModel




def aov(y, X, C, Q, gg=False, _Xeff=None):
	model    = GeneralLinearModel()
	model.set_design_matrix( X )
	model.set_contrast_matrix( C )
	model.set_variance_model( Q )
	fit      = model.fit( y )
	fit.estimate_variance()
	fit.calculate_effective_df( _Xeff )
	fit.calculate_f_stat()
	if gg:
		fit.greenhouse_geisser()
	f,df     = fit.f, fit.df
	if fit.dvdim==1:
		p  = scipy.stats.f.sf(float(f), df[0], df[1])
	else:
		fwhm    = rft1d.geom.estimate_fwhm( fit.e )
		p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
	return f, df, p, model



def anova1(y, A, equal_var=False):
	from . designs import ANOVA1
	design   = ANOVA1( A )
	Q        = design.get_variance_model( equal_var=equal_var )
	return aov(y, design.X, design.C, Q)
	
	
def anova1rm(y, A, SUBJ, equal_var=False, gg=True):
	from . designs import ANOVA1RM
	design   = ANOVA1RM( A, SUBJ )
	Q        = design.get_variance_model( equal_var=equal_var )
	return aov(y, design.X, design.C, Q, gg=True, _Xeff= design.X[:,:-1] )  # "design.X[:,:-1]" is a hack;  there must be a different way
	# model    = GeneralLinearModel()
	# model.set_design_matrix( design.X )
	# model.set_contrast_matrix( design.C )
	# model.set_variance_model( Q )
	# fit      = model.fit( y )
	# fit.estimate_variance()
	# fit.calculate_effective_df(  design.X[:,:-1]  )   # "design.X[:,:-1]" is a hack;  there must be a different way
	# fit.calculate_f_stat()
	# if gg:
	# 	fit.greenhouse_geisser()
	# f,df   = fit.f, fit.df
	# if fit.dvdim==1:
	# 	p  = scipy.stats.f.sf(float(f), df[0], df[1])
	# else:
	# 	fwhm    = rft1d.geom.estimate_fwhm( fit.e )
	# 	p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
	# return f, df, p, model

# def anova1(y, A, equal_var=False):
# 	from . designs import ANOVA1
# 	design   = ANOVA1( A )
# 	Q        = design.get_variance_model( equal_var=equal_var )
# 	model    = GeneralLinearModel()
# 	model.set_design_matrix( design.X )
# 	model.set_contrast_matrix( design.C )
# 	model.set_variance_model( Q )
# 	fit      = model.fit( y )
# 	fit.estimate_variance()
# 	fit.calculate_effective_df()
# 	fit.calculate_f_stat()
# 	f,df     = fit.f, fit.df
# 	if fit.dvdim==1:
# 		p  = scipy.stats.f.sf(float(f), df[0], df[1])
# 	else:
# 		fwhm    = rft1d.geom.estimate_fwhm( fit.e )
# 		p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
# 	return f, df, p, model


# def anova1rm(y, A, SUBJ, equal_var=False, gg=True):
# 	from . designs import ANOVA1RM
# 	design   = ANOVA1RM( A, SUBJ )
# 	Q        = design.get_variance_model( equal_var=equal_var )
# 	model    = GeneralLinearModel()
# 	model.set_design_matrix( design.X )
# 	model.set_contrast_matrix( design.C )
# 	model.set_variance_model( Q )
# 	fit      = model.fit( y )
# 	fit.estimate_variance()
# 	fit.calculate_effective_df(  design.X[:,:-1]  )   # "design.X[:,:-1]" is a hack;  there must be a different way
# 	fit.calculate_f_stat()
# 	if gg:
# 		fit.greenhouse_geisser()
# 	f,df   = fit.f, fit.df
# 	if fit.dvdim==1:
# 		p  = scipy.stats.f.sf(float(f), df[0], df[1])
# 	else:
# 		fwhm    = rft1d.geom.estimate_fwhm( fit.e )
# 		p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
# 	return f, df, p, model



# def anova1(y, A, equal_var=False):
# 	model  = OneWayANOVAModel(y, A)
# 	model.build_variance_model( equal_var=equal_var )
# 	model.fit()
# 	model.estimate_variance()
# 	model.calculate_effective_df()
# 	model.calculate_f_stat()
# 	f,df   = model.f, model.df
#
# 	if isinstance(f, float):
# 		p  = scipy.stats.f.sf(f, df[0], df[1])
# 	else:
# 		fwhm    = rft1d.geom.estimate_fwhm( model.e )
# 		p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
#
# 	return f, df, p, model



# def anova1rm(y, A, SUBJ, equal_var=False, gg=True):
# 	model  = OneWayRMANOVAModel(y, A, SUBJ)
# 	model.build_variance_model( equal_var=equal_var )
# 	model.fit()
# 	model.estimate_variance(  )
# 	model.calculate_effective_df()
# 	model.calculate_f_stat()
# 	if gg:
# 		model.greenhouse_geisser()
#
# 	f,df,V   = model.f, model.df, model.V
#
# 	if isinstance(f, float):
# 		p  = scipy.stats.f.sf(f, df[0], df[1])
# 	else:
# 		fwhm    = rft1d.geom.estimate_fwhm( model.e )
# 		p       = rft1d.f.sf(f.max(), df, y.shape[1], fwhm)
# 	return f, df, p, model