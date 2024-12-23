

import numpy as np
import scipy

from ..theoretical_profiles import AbstractBaseProfile
from .composite_utils import get_required_params_num

class MultiProfiles:
    
    
    
    def __init__(self,):
        self._profiles = []
        self._coefs = []
        self._panums = []
        
    def add_profile(self,profile):
        if not issubclass(profile,AbstractBaseProfile):
            raise TypeError('Only theoretical profiles are allowed')
        self._profiles.append(profile)
        self._coefs.append(1.)
        self._panums.append(get_required_params_num(profile))
    
    def parameter_bounds(self,radial_data,profile_data):
        lower_bounds = []
        upper_bounds = []
        for i in self._profiles:
            lower, upper = i.parameter_bounds(radial_data, profile_data)
            lower_bounds = lower_bounds +lower
            upper_bounds = upper_bounds +upper
            
        lower_bounds = lower_bounds + list(np.zeros(len(self._coefs)))
        upper_bounds = upper_bounds + list(np.ones(len(self._coefs)))
        
        return (lower_bounds,upper_bounds)
    
    def __call__(self, radius, *args):
        
        parameter = args if args else tuple(self._paall)
        
        pr = 0 if (isinstance(radius,int) or isinstance(radius,float)) else np.zeros(len(radius))
        for i in range(len(self._profiles)):
            ind_s = int(np.sum(self._panums[:i]))
            ind_e= int(np.sum(self._panums[:i+1]))
            pr = pr + self._profiles[i](*parameter[ind_s:ind_e])(radius)*self._coefs[i]
        return pr
        
    def set_parameter(self,*args):
        self._paall= []
        for i in args:
            self._paall.append(i)
        
    def jacobian(self, radius, *args):
        parameter = args if args else tuple(self._paall)
        
        jacobian_all=[]
        coef_jacobian = []
        for i in range(len(self._profiles)):
            ind_s = int(np.sum(self._panums[:i]))
            ind_e= int(np.sum(self._panums[:i+1]))
            jacobian_all.append(np.transpose(self._profiles[i](*parameter[ind_s:ind_e]).jacobian(radius)*self._coefs[i]))
            coef_jacobian.append(self._profiles[i](*parameter[ind_s:ind_e])(radius))
        return np.transpose(np.vstack([np.vstack(jacobian_all),np.vstack(coef_jacobian)]))
    
    def enclosed_mass(self, radius, *args):
        parameter = args if args else tuple(self._paall)
        mass = 0 if (isinstance(radius,int) or isinstance(radius,float)) else np.zeros(len(radius))
        for i in range(len(self._profiles)):
            ind_s = int(np.sum(self._panums[:i]))
            ind_e= int(np.sum(self._panums[:i+1]))
            mass = mass + self._profiles[i](*parameter[ind_s:ind_e]).enclosed_mass(radius)*self._coefs[i]
        return mass
    
    def components_enclosed_mass(self,radius,*args):
        parameter = args if args else tuple(self._paall)
        mass = []
        for i in range(len(self._profiles)):
            ind_s = int(np.sum(self._panums[:i]))
            ind_e= int(np.sum(self._panums[:i+1]))
            mass.append(self._profiles[i](*parameter[ind_s:ind_e]).enclosed_mass(radius)*self._coefs[i])
        return tuple(mass)
        
    def components_profile(self,radius,*args):
        parameter = args if args else tuple(self._paall)
        
        pr = []
        for i in range(len(self._profiles)):
            ind_s = int(np.sum(self._panums[:i]))
            ind_e= int(np.sum(self._panums[:i+1]))
            pr.append(self._profiles[i](*parameter[ind_s:ind_e])(radius)*self._coefs[i])
        return tuple(pr)
    
    
    def __repr__(self):
        contain = '||'
        for i in self._profiles:
            contain = contain + i.__class__.__name__+'|'
        contain = contain + '|'
        return "<" + contain +">"
        
    def fit(self, radial_data, profile_data, profile_err=None, use_analytical_jac=True, guess=None, verbose=0,
            return_profile = False, **kwargs):
        """Fit the given profile using a least-squares method.

        Parameters
        ----------

        radial_data : array_like
            The central radius of the bins in which the profile data is measured

        profile_data : array_like
            The profile density values

        profile_err : array_like, optional
            The error on the profile data

        use_analytical_jac : bool
            Whether to use the analytical jacobian of the profile function. If False, finite differencing is used.

        guess : array_like, optional
            An initial guess for the parameters of the profile. If None, the initial guess is taken to be all ones,
            according to the underlying ``scipy.optimize.curve_fit`` function.

        verbose : int
            The verbosity level to pass to the underlying ``scipy.optimize.curve_fit`` function.

        return_profile : bool
            Whether to return the profile object or just the parameters

        Returns
        -------

        fitted_profile : array_like | AbstractBaseProfile
            If return_profile is True, the fitted profile object. Otherwise, the fitted parameters.

        cov : array_like
            The covariance matrix of the fit. The diagonal elements are the variance of the parameters.

        """

        import scipy.optimize as so
        from collections import Counter

        # Check data is not corrupted. Some are likely check in curve-fit already
        if np.isnan(radial_data).any() or np.isnan(profile_data).any():
            raise RuntimeError("Provided data contains NaN values")

        if np.count_nonzero(radial_data) != radial_data.size or np.count_nonzero(profile_data) != profile_data.size:
            raise RuntimeError("Provided data contains zeroes. This is likely to make the fit fail.")

        if radial_data.size != profile_data.size != profile_err.size:
            raise RuntimeError("Provided data arrays do not match in shape")

        if use_analytical_jac:
            jac = self.jacobian
        else:
            jac = '3-point'

        lower_bounds, upper_bounds = self.parameter_bounds(radial_data, profile_data)

        def profile_wrapper(radius, *args):
            return self(radius, *args)


        # the same type profile,  with different initial guesses
        count = Counter(self._profiles)
        basecoun = {i: 1. for i in count}
        
        if guess is None:
            guess = (np.asarray(upper_bounds) + np.asarray(lower_bounds)) / 2.0
            for i in range(len(self._profiles)):
                if count[self._profiles[i]]>1:
                    ind_s = int(np.sum(self._panums[:i]))
                    ind_e= int(np.sum(self._panums[:i+1]))
                    guess[ind_s:ind_e] = guess[ind_s:ind_e]*2*(basecoun[self._profiles[i]]/(1+count[self._profiles[i]]))
                    basecoun[self._profiles[i]] = basecoun[self._profiles[i]] + 1

        ftol=kwargs.get('ftol',1e-10)
        xtol=kwargs.get('xtol',1e-10)
        gtol=kwargs.get('gtol',1e-10)
        x_scale=kwargs.get('x_scale',1.0)
        loss=kwargs.get('loss','linear')
        f_scale=kwargs.get('f_scale',1.0)
        max_nfev=kwargs.get('max_nfev',len(guess)*1000)         #origin 100*n, changed to 1000*n
        diff_step=kwargs.get('diff_step',None)
        tr_solver=kwargs.get('tr_solver',None)
        try:
            parameters, cov = so.curve_fit(profile_wrapper,
                                           radial_data,
                                           profile_data,
                                           sigma=profile_err,
                                           p0=guess,
                                           bounds=(lower_bounds, upper_bounds),
                                           check_finite=True,
                                           jac=jac,
                                           method='trf',
                                           ftol=ftol,
                                           xtol=xtol,
                                           gtol=gtol,
                                           x_scale=x_scale,
                                           loss=loss,
                                           f_scale=f_scale,
                                           max_nfev=max_nfev,
                                           diff_step=diff_step,
                                           tr_solver=tr_solver,
                                           verbose=verbose)
        except so.OptimizeWarning as w:
            raise RuntimeError(str(w))

        if (guess is None and any(parameters == np.ones(parameters.shape))) or any(parameters == guess):
            raise RuntimeError("Fitted parameters are equal to their initial guess. This is likely a failed fit.")

        self.set_parameter(*parameters)
        if return_profile:
            return self(radial_data), cov
        else:
            return parameters, cov
