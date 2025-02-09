"""
a modified version from https://pynbody.readthedocs.io/latest/_modules/pynbody/analysis/theoretical_profiles.html

Implements functional forms of known density profiles and code to fit them to simulation data.

At present only the NFW profile is implemented, but the code is designed to be easily extensible to other profiles.


pynbody profiles: AbstractBaseProfile,
AbstractBaseProfile: add logfit, make the fit more robust in some cases
"""

import abc

import numpy as np
import scipy


class AbstractBaseProfile(abc.ABC):
    """
    Represents an analytic profile of a halo, and provides a method to fit the profile to data.

    The class is organised a dictionary, i.e. profile parameters of a given instance can be accessed through
    ``profile['...']`` and the available parameters are listed in ``profile.keys()``. The parameters are set at
    initialisation and cannot be changed afterwards.

    To define a new profile, create a new class inheriting from this base class and define your own profile_functional()
    method. The static version can be handy to avoid having to create and object every time.
    As a example, the NFW functional is implemented.

    A generic fitting function is provided (:meth:`fit`). Given a binned quantity as a function of radius, it
    uses least-squares to fit the given functional form to the data.

    """
    def __init__(self):
        self._parameters = dict()
        self.__ndim = 3
        self._rmin = 0.
    @classmethod
    @abc.abstractmethod
    def parameter_bounds(self, r_values, rho_values):
        """Return bounds on the parameter values for the profile fit

        Parameters
        ----------

        r_values : array_like
            The radii at which the profile is measured

        rho_values : array_like
            The density values of the profile


        Returns
        -------

        bounds : tuple[array_like]
            A 2-tuple containing lower and upper bounds respectively for the parameters of the profile

        """
        pass
    @property
    def ndim(self):
        return self.__ndim
    @ndim.setter
    def ndim(self,ndim):
        if ndim in [1,2,3]:
            self.__ndim=ndim
            return
        raise TypeError('Only 1,2,3')
    
    @abc.abstractmethod
    def logarithmic_slope(self, radius):
        """Return the logarithmic slope of the profile, d ln rho / d ln r, at a given radius"""
        pass

    @abc.abstractmethod
    def enclosed_mass(self, radius):
        """Return the mass, M(r), enclosed within a given radius"""
        pass
    
    def Integrate(self, radius, **kwargs):
        '''Return the mass, determined by dim and radius '''
        from scipy.integrate import quad
        ndim = kwargs.get('ndim',self.__ndim)
        rmin = kwargs.get('rmin',self.__ndim)
        def d_mass_1D(radius):
            return self(radius)
        def d_mass_2D(radius):
            return self(radius)*np.pi*2*radius
        def d_mass_3D(radius):
            return self(radius)*np.pi*4*radius**2
        func = {1:d_mass_1D,2:d_mass_2D,3:d_mass_3D}
        
        if isinstance(radius,int) or isinstance(radius,float):
            mass,err = quad(func[ndim],rmin,radius)
            return mass
        if hasattr(radius,'__iter__'):
            return np.array([quad(func[ndim],rmin,i) for i in radius])[:,0]
        raise TypeError('Unknown type')

    @classmethod
    def fit(cls, radial_data, profile_data, profile_err=None, use_analytical_jac=True, guess=None, verbose=0,
            logfit = True,return_profile = True,**kwargs):
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
        
        logfit: bool
            Whether to fit data in the log space 
        
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

        # Check data is not corrupted. Some are likely check in curve-fit already
        if np.isnan(radial_data).any() or np.isnan(profile_data).any():
            raise RuntimeError("Provided data contains NaN values")

        if np.count_nonzero(radial_data) != radial_data.size or np.count_nonzero(profile_data) != profile_data.size:
            raise RuntimeError("Provided data contains zeroes. This is likely to make the fit fail.")

        if radial_data.size != profile_data.size != profile_err.size:
            raise RuntimeError("Provided data arrays do not match in shape")
        if (radial_data<0).any() and logfit:
            raise RuntimeError("Provided data contains negative values, should set logfit=False")

        if use_analytical_jac:
            def jacobian_wrapper(radius, *args):
                return cls(*args).jacobian(radius)
            if logfit:                              
                def jacobian_wrapper(radius, *args):
                    return np.einsum('ij,i->ij',cls(*args).jacobian(radius),1/cls(*args)(radius))
            jac = jacobian_wrapper
        else:
            jac = '3-point'

        lower_bounds, upper_bounds = cls.parameter_bounds(radial_data, profile_data)

        def profile_wrapper(radius, *args):
            return cls(*args)(radius)
        if logfit:
            def profile_wrapper(radius, *args):
                return np.log( cls(*args)(radius))

        if guess is None:
            guess = (np.asarray(upper_bounds) + np.asarray(lower_bounds)) / 2.0

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
            if logfit:
                profile_data= np.log(profile_data)
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
        if return_profile:
            return cls(*parameters), cov
        else:
            return parameters, cov

    def __getitem__(self, item):
        return self._parameters.__getitem__(item)

    def __setitem__(self, key, value):
        raise KeyError('Cannot change a parameter from the profile once set')

    def __delitem__(self, key):
        raise KeyError('Cannot delete a parameter from the profile once set')

    def __repr__(self):
        return "<" + self.__class__.__name__ + str(list(self.keys())) + ">"
    def chi2(self,radial_data,profile_data,profile_error=None,mode='Model'):
        radial_data = radial_data.view(np.ndarray)
        profile_data = profile_data.view(np.ndarray)
        w = profile_error
        if isinstance(w,type(None)):
            w = np.sqrt(np.abs(profile_data))
        if mode =='Model':
            if not isinstance(w,type(None)):
                w = w*np.sqrt(np.abs(self(radial_data)))
            else:
                w = np.sqrt(np.abs(self(radial_data)))
            return np.sum((self(radial_data)-profile_data)**2/w**2)
        if mode == 'Data':
            if not isinstance(w,type(None)):
                w = w*np.sqrt(np.abs(profile_data))
            else:
                w = np.sqrt(np.abs(profile_data))
            return np.sum((self(radial_data)-profile_data)**2/w**2)
        if mode == 'Cash':
            return 2*np.sum((self(radial_data)-profile_data*np.log(self(radial_data))))
        if mode == 'PMLR':
            return 2*np.sum((self(radial_data)-profile_data*np.log(self(radial_data))
                               +profile_data*np.log(profile_data)-profile_data))
    def AIC(self,radial_data,profile_data,**kwargs):
        profile_error=kwargs.get('profile_error',None)
        mode=kwargs.get('mode','Model')
        chi2 = self.chi2(radial_data=radial_data,profile_data=profile_data,profile_error=profile_error,mode=mode)
        k = len(self.keys())
        n = len(radial_data)
        return chi2+2*k+2*k*(k+1)/(n-k-1)
    
    def BIC(self,radial_data,profile_data,**kwargs):
        profile_error=kwargs.get('profile_error',None),
        mode=kwargs.get('mode','Model')
        chi2 = self.chi2(radial_data=radial_data,profile_data=profile_data,profile_error=profile_error,mode=mode)
        k = len(self.keys())
        n = len(radial_data)
        return chi2+k*np.log(n)
        
    def keys(self):
        """Return the keys of the profile parameters"""
        return list(self._parameters.keys())

    def __contains__(self, item):
        return item in self._parameters
    
# from pyimfit https://github.com/perwin/pyimfit/blob/master/pyimfit/imfit_funcs.py
# Notes on the different "incomplete gamma functions"
# http://mpmath.org/doc/current/functions/expintegrals.html?highlight=gammainc#incomplete-gamma-functions
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainc.html
def gammainc_lower_scipy( z, a, b):
    a=0
    return scipy.special.gamma(z) * scipy.special.gammainc(z, b)
try:
    from mpmath import besselk as BesselK   # type: ignore
    from mpmath import gamma as Gamma
    from mpmath import gammainc as GammaInc
except ImportError:
    from scipy.special import kv as BesselK
    from scipy.special import gamma as Gamma
    GammaInc = gammainc_lower_scipy
    
class Latex_print():
    def __init__(self,text):
        self.t = text
    def _repr_latex_(self):
        return f"$${self.t.strip('$')}$$"
