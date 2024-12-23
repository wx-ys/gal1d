"""
a modified version from https://pynbody.readthedocs.io/latest/_modules/pynbody/analysis/theoretical_profiles.html

Implements functional forms of known density profiles and code to fit them to simulation data.

At present only the NFW profile is implemented, but the code is designed to be easily extensible to other profiles.


pynbody profiles: AbstractBaseProfile,
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

    @abc.abstractmethod
    def logarithmic_slope(self, radius):
        """Return the logarithmic slope of the profile, d ln rho / d ln r, at a given radius"""
        pass

    @abc.abstractmethod
    def enclosed_mass(self, radius):
        """Return the mass, M(r), enclosed within a given radius"""
        pass

    @classmethod
    def fit(cls, radial_data, profile_data, profile_err=None, use_analytical_jac=True, guess=None, verbose=0,
            return_profile = True, **kwargs):
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

        # Check data is not corrupted. Some are likely check in curve-fit already
        if np.isnan(radial_data).any() or np.isnan(profile_data).any():
            raise RuntimeError("Provided data contains NaN values")

        if np.count_nonzero(radial_data) != radial_data.size or np.count_nonzero(profile_data) != profile_data.size:
            raise RuntimeError("Provided data contains zeroes. This is likely to make the fit fail.")

        if radial_data.size != profile_data.size != profile_err.size:
            raise RuntimeError("Provided data arrays do not match in shape")

        if use_analytical_jac:
            def jacobian_wrapper(radius, *args):
                return cls(*args).jacobian(radius)
            jac = jacobian_wrapper
        else:
            jac = '3-point'

        lower_bounds, upper_bounds = cls.parameter_bounds(radial_data, profile_data)

        def profile_wrapper(radius, *args):
            return cls(*args)(radius)

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

    def keys(self):
        """Return the keys of the profile parameters"""
        return list(self._parameters.keys())

    def __contains__(self, item):
        return item in self._parameters