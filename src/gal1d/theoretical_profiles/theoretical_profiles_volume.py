"""

pynbody profiles: NFWProfile

added profiles: SersicProfile, ExponentialProfile
"""


import numpy as np
import scipy

from .base_utils import AbstractBaseProfile


class NFWProfile(AbstractBaseProfile):
    """Represents a Navarro-Frenk-White (NFW) profile."""

    def __init__(self, density_scale_radius=None, scale_radius=None, halo_radius=None, concentration=None,
                 halo_mass=None):
        """Represents a Navarro-Frenk-White (NFW) profile.

        The profile can then be initialised through one of the following combination of parameters:

        * *scale_radius*, *density_scale_radius* and optionally *halo_radius*;
        * *halo_radius*, *concentration* and *density_scale_radius*;
        * *halo_radius*, *concentration* and *halo_mass*.

        From one mode of initialisation, the derived parameters of the others are calculated, e.g. if you initialise
        with halo_mass + concentration, the scale_radius and central density will be derived. The exception is if
        you initialise with *scale_radius* + *density_scale_radius* without *halo_radius*.

        Units may be passed into the parameters by using scalar arrays.

        Parameters
        ----------

        scale_radius : float | array-like, optional
            The radius at which the slope is equal to -2

        density_scale_radius : float | array-like, optional
            1/4 of density at r=rs (normalisation).

        halo_mass : float | array-like, optional
            The mass enclosed inside the outer halo radius

        halo_radius : float | array-like
            The outer boundary of the halo (r200m, r200c, rvir ... depending on definitions)

        concentration : float | array-like, optional
            The outer_radius / scale_radius

        """

        super().__init__()

        if scale_radius is not None and density_scale_radius is not None and concentration is None and halo_mass is None:
            self._parameters['scale_radius'] = scale_radius
            self._parameters['density_scale_radius'] = density_scale_radius
            if halo_radius is not None:
                self._parameters['halo_radius'] = halo_radius
                self._parameters['concentration'] = halo_radius / scale_radius
                self._parameters['halo_mass'] = self.enclosed_mass(halo_radius)
        elif (halo_radius is not None and concentration is not None and density_scale_radius is not None
              and halo_mass is None):
            self._parameters['halo_radius'] = halo_radius
            self._parameters['concentration'] = concentration
            self._parameters['density_scale_radius'] = density_scale_radius
            self._parameters['scale_radius'] = halo_radius / concentration
            self._parameters['halo_mass'] = self.enclosed_mass(halo_radius)
        elif (halo_radius is not None and concentration is not None and halo_mass is not None
              and density_scale_radius is None):
            self._parameters['halo_radius'] = halo_radius
            self._parameters['concentration'] = concentration
            self._parameters['scale_radius'] = halo_radius / concentration
            self._parameters['halo_mass'] = halo_mass
            self._parameters['density_scale_radius'] = self._derive_central_overdensity()
        else:
            raise ValueError("Invalid combination of parameters for initializing NFWProfile.")

    @classmethod
    def parameter_bounds(cls, r_values, rho_values):
        profile_lower_bound = np.amin(rho_values)
        profile_upper_bound = np.amax(rho_values)

        radial_lower_bound = np.amin(r_values)
        radial_upper_bound = np.amax(r_values)

        return ([profile_lower_bound, radial_lower_bound], [profile_upper_bound, radial_upper_bound])

    def jacobian(self, radius):
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']

        d_scale_radius = density_scale_radius * (3 * radius / scale_radius + 1) / (
                    radius * (1 + radius / scale_radius) ** 3)
        d_central_density = 1 / ((radius / scale_radius) * (1 + radius / scale_radius) ** 2)
        return np.transpose([d_central_density, d_scale_radius])

    def __call__(self, radius):
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        return density_scale_radius / ((radius / scale_radius) * (1.0 + (radius / scale_radius)) ** 2)

    def enclosed_mass(self, radius):
        # Eq 7.139 in M vdB W
        radius = np.asarray(radius)
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        x = radius / scale_radius
        coef = self._integral(x)
        return density_scale_radius * scale_radius ** 3 * coef
               
    def enclosed_mass_2d(self, radius):
        radius = np.asarray(radius)
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        x = radius / scale_radius
        coef = self._g(x)
        return 4 * density_scale_radius * scale_radius * radius**2 * coef / x**2 * np.pi
    
    def density_2d(self,radius):
        radius = np.asarray(radius)
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        x = radius / scale_radius
        coef = self._F(x)
        return 2 * density_scale_radius * scale_radius * coef
    
    def potential(self,radius):
        radius = np.asarray(radius)
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        x = radius / scale_radius
        coef = self._h(x)
        return 2 * density_scale_radius * scale_radius**3 * coef
        
    
    def _derive_concentration(self):
        return self._parameters['halo_radius'] / self._parameters['scale_radius']

    def _derive_scale_radius(self):
        return self._parameters['halo_radius'] / self._parameters['concentration']

    def _derive_central_overdensity(self):
        return self._parameters['halo_mass'] / (self._parameters['scale_radius']**3 *
                                                self._integral(self._parameters['concentration']))

    def logarithmic_slope(self, radius):
        radius = np.asarray(radius)
        scale_radius = self._parameters['scale_radius']
        return - (1.0 + 3.0 * radius / scale_radius) / (1.0 + radius / scale_radius)

    @staticmethod
    def _integral(x):
        return 4 * np.pi * (np.log(1.0 + x) - x / (1.0 + x))
    
    @staticmethod
    def _g(X):
        """Analytic solution of integral for NFW profile to compute deflection angel and
        gamma.

        :param X: R/Rs
        :type X: float >0
        ----
        from https://lenstronomy.readthedocs.io/en/latest/_modules/lenstronomy/LensModel/Profiles/nfw.html
        """
        c = 0.000001
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(c, X)
                a = np.log(x / 2.0) + 1 / np.sqrt(1 - x**2) * np.arccosh(1.0 / x)
            elif X == 1:
                a = 1 + np.log(1.0 / 2.0)
            else:  # X > 1:
                a = np.log(X / 2) + 1 / np.sqrt(X**2 - 1) * np.arccos(1.0 / X)
        else:
            a = np.empty_like(X)
            X[X <= c] = c
            x = X[X < 1]
            a[X < 1] = np.log(x / 2.0) + 1 / np.sqrt(1 - x**2) * np.arccosh(1.0 / x)
            a[X == 1] = 1 + np.log(1.0 / 2.0)
            x = X[X > 1]
            a[X > 1] = np.log(x / 2) + 1 / np.sqrt(x**2 - 1) * np.arccos(1.0 / x)
        return a
    
    @staticmethod
    def _F(X):
        """Analytic solution of the projection integral.

        :param X: R/Rs
        :type X: float >0
        ----
        from https://lenstronomy.readthedocs.io/en/latest/_modules/lenstronomy/LensModel/Profiles/nfw.html
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1 and X > 0:
                a = (1 / (X**2 - 1) * (1 - 2 / np.sqrt(1 - X**2) * np.arctanh(np.sqrt((1 - X) / (1 + X)))))
            elif X == 1:
                a = 1.0 / 3
            elif X > 1:
                a = (1 / (X**2 - 1) * (1 - 2 / np.sqrt(X**2 - 1) * np.arctan(np.sqrt((X - 1) / (1 + X)))))
            else:  # X == 0:
                c = 0.0000001
                a = (1 / (-1) * (1 - 2 / np.sqrt(1) * np.arctanh(np.sqrt((1 - c) / (1 + c)))))
        else:
            a = np.empty_like(X)
            x = X[(X < 1) & (X > 0)]
            a[(X < 1) & (X > 0)] = (1 / (x**2 - 1) * (1 - 2 / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))))
            a[X == 1] = 1.0 / 3.0
            x = X[X > 1]
            a[X > 1] = (1 / (x**2 - 1) * (1 - 2 / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x)))))
            # a[X>y] = 0
            c = 0.0000001
            a[X == 0] = (1 / (-1) * (1 - 2 / np.sqrt(1) * np.arctanh(np.sqrt((1 - c) / (1 + c)))))
        return a
    @staticmethod
    def _h(X):
        """Analytic solution of integral for NFW profile to compute the potential.

        :param X: R/Rs
        :type X: float >0
        ------
        from https://lenstronomy.readthedocs.io/en/latest/_modules/lenstronomy/LensModel/Profiles/nfw.html
        """
        c = 0.000001
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(0.001, X)
                a = np.log(x / 2.0) ** 2 - np.arccosh(1.0 / x) ** 2
            else:  # X >= 1:
                a = np.log(X / 2.0) ** 2 + np.arccos(1.0 / X) ** 2
        else:
            a = np.empty_like(X)
            X[X <= c] = 0.000001
            x = X[X < 1]
            a[X < 1] = np.log(x / 2.0) ** 2 - np.arccosh(1.0 / x) ** 2
            x = X[X >= 1]
            a[X >= 1] = np.log(x / 2.0) ** 2 + np.arccos(1.0 / x) ** 2
        return a


    
class GNFWProfile(AbstractBaseProfile):
    '''Represents a Generalized NFW profile profile.'''
    
    BOUND = {}
    BOUND['density_scale_radius'] = None
    BOUND['scale_radius'] = None
    BOUND['gamma'] = [0,3]
    
    # numerical limit for minimal radius
    _s = 0.001  
    
    #if True, the numerical integral is performed with the trapezoidal rule, otherwise with ~scipy.integrate.quad
    _trapezoidal_integration = False 
    
    _integration_steps = 1000
    def __init__(self, density_scale_radius=None, scale_radius=None, gamma=None):
        '''
        Represents a Generalized NFW profile profile.
        Parameters
        ----------
        density_scale_radius : float | array-like, 
            1/4 of density at r=rs (normalisation).

        scale_radius : float | array-like, 

        gamma : float | array-like, 0 - 3
            inner slope
            
        # Kravtsov 1998
        rho(r) = rho0 / [ (r / r_s)^(gamma) * (1 + (r / r_s))^(3-gamma) ]  
        '''
        super().__init__()
        self._parameters['density_scale_radius']=density_scale_radius
        self._parameters['scale_radius']=scale_radius
        self._parameters['gamma']=gamma
        
    @classmethod
    def parameter_bounds(cls, r_values, rho_values):
        if cls.BOUND['density_scale_radius']:
            density_lower_bound = cls.BOUND['density_scale_radius'][0]
            density_upper_bound = cls.BOUND['density_scale_radius'][1]
        else:
            density_lower_bound = np.amin(rho_values)
            density_upper_bound = np.amax(rho_values)

        if cls.BOUND['scale_radius']:
            radial_lower_bound = cls.BOUND['scale_radius'][0]
            radial_upper_bound = cls.BOUND['scale_radius'][1]
        else:
            radial_lower_bound = np.amin(r_values)
            radial_upper_bound = np.amax(r_values)
        
        gamma_lower_bound = cls.BOUND['gamma'][0]
        gamma_upper_bound = cls.BOUND['gamma'][1]
        
        return ([density_lower_bound, radial_lower_bound, gamma_lower_bound], 
                [density_upper_bound, radial_upper_bound, gamma_upper_bound])
    
    def jacobian(self, radius):
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        gamma = self._parameters['gamma']
        coef1 = np.power((radius/scale_radius),gamma)
        coef2 = np.power((1+radius/scale_radius),3 - gamma)
        d_density_scale_radius = 1/ (coef1 * coef2)
        
        temp_coef1 = density_scale_radius*np.power((scale_radius+radius),gamma)*(
            gamma*scale_radius**3 + 3*radius*scale_radius**2)
        temp_coef2 = np.power(radius,gamma)*(
            scale_radius**4+ 4*radius*scale_radius**3 + 6*np.power(radius,2)*scale_radius**2
            + 4*np.power(radius,3)*scale_radius + np.power(radius,4))
        d_scale_radius = temp_coef1/temp_coef2
        
        temp_coef1 = np.power((radius+scale_radius)/scale_radius,gamma)
        temp_coef2 = (scale_radius**3*density_scale_radius*np.log((radius+scale_radius)/scale_radius)
                      - scale_radius**3 * density_scale_radius *np.log(radius/scale_radius))
        temp_coef3 = coef1 * (np.power(radius,3)+3*scale_radius*np.power(radius,2)+3*scale_radius**2*radius+scale_radius**3)
        d_gamma = temp_coef1*temp_coef2/temp_coef3
        
        
        return np.transpose([d_density_scale_radius, d_scale_radius,d_gamma])
    
    def __call__(self,radius):
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        gamma = self._parameters['gamma']
        coef1 = np.power((radius/scale_radius),gamma)
        coef2 = np.power((1+radius/scale_radius),3 - gamma)
        
        return density_scale_radius/ (coef1 * coef2)
    
    def enclosed_mass(self, radius):
        radius = np.asarray(radius)
        from scipy.special import hyp2f1
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        gamma = self._parameters['gamma']
        
        M_0 = 4 * np.pi * density_scale_radius * scale_radius**3 / (3 - gamma)
        x = radius / scale_radius
        return M_0* x ** (3 - gamma) * hyp2f1(3 - gamma, 3 - gamma, 4 - gamma, -x)
    
    def density_2d(self,radius):
        radius = np.asarray(radius)
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        gamma = self._parameters['gamma']
        x = radius/scale_radius
        x = np.clip(x, self._s, np.max(x))
        kappa_integrand = lambda y, x, gamma: (y + x) ** (gamma - 4) * (1 - np.sqrt(1 - y**2))
        
        integral = self._integral(kappa_integrand, x)
        
        kappa = 2 * density_scale_radius * scale_radius * x ** (1 - gamma) * ((1 + x) ** (gamma - 3) + (3 - gamma) * integral)
        
        return kappa
    
    def logarithmic_slope(self, radius):
        radius = np.asarray(radius)
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        gamma = self._parameters['gamma']
        
        coef1 =scale_radius**3 * density_scale_radius * np.power(radius+scale_radius,gamma)/np.power(radius,gamma)
        coef2 = np.power(radius,3)+3*scale_radius*np.power(radius,2)+3*scale_radius**2 *radius +scale_radius**3
        coef3 = (3*radius+scale_radius*gamma)/(radius+scale_radius)
        
        return np.log(coef1/coef2)-coef3
        
    def _integral(self, func, x):
        gamma = self._parameters['gamma']
        if self._trapezoidal_integration:
            steps = self._integration_steps
            return self._trapezoidal_integrate(func, x, gamma, steps)
        else:
            return self._quad_integrate(func, x, gamma)
        
    @staticmethod
    def _trapezoidal_integrate(func, x, gamma, steps):
        y = np.linspace(1e-10, 1 - 1e-10, steps)
        dy = y[1] - y[0]

        weights = np.ones(steps)
        weights[0] = 0.5
        weights[-1] = 0.5
        
        if isinstance(x, int) or isinstance(x, float):
            integral = np.sum(func(y, x, gamma) * dy * weights)
        else:
            x_flat = x.flatten()
            ys = np.repeat(y[:, np.newaxis], len([x_flat]), axis=1)

            integral = np.sum(func(ys, x_flat, gamma) * dy * weights[:, np.newaxis], axis=0)
            integral = integral.reshape(x.shape)
        return integral
    
    @staticmethod
    def _quad_integrate(func, x, gamma):
        from scipy.integrate import quad
        if isinstance(x, int) or isinstance(x, float):
            integral = quad(func, a=0, b=1, args=(x, gamma))[0]
        else:
            integral = np.zeros_like(x)
            for i in range(len(x)):
                integral[i] = quad(func, a=0, b=1, args=(x[i], gamma))[0]
        return integral
    

class DoublePowerLawProfile(AbstractBaseProfile):
    '''a 'broken' or double power-law profile'''
    
    BOUND = {}
    BOUND['density_scale_radius'] = None
    BOUND['scale_radius'] = None
    BOUND['alpha'] = [0,10]
    BOUND['beta'] = [0,10]
    BOUND['gamma'] = [0,3]
    
    def __init__(self,density_scale_radius,scale_radius,alpha,beta,gamma):
        '''Represents a double power-law profile, first proposed by Hernquist (1990) and Zhao (1996)
        use the formula in An & Zhao (2013) eq 1.
        density_scale_radius: rho0, scale_radius: a
        rho(r) = rho0 * (a)^(beta) / [(r^alpha) * (a^gamma + r^gamma)^[(beta-alpha)/gamma]
        
        Parameters
        ----------
        scale_radius : float | array-like, optional
            The radius at which the slope is equal to -(alpha + beta)
        
        density_scale_radius : float | array-like, optional
        
        alpha: float | array-like, optional
        
        beta: float | array-like, optional
        
        gamma: float | array-like, optional
        ----------
        (alpha=0, beta=5, gamma=2): 
            the Schuster (1884)-Plummer (1911) sphere, an index-5 polytrope
            
        (alpha=2, beta=4, gamma=1): 
            the Jaffe (1983) model
            
        (alpha=1, beta=4, gamma=1): 
            the Hernquist (1990) model
            
        (alpha=1, beta=3, gamma=1): 
            the NFW profile
            
        (alpha=7/9, beta=31/9, gamma=4/9): 
            the isotropic analytical solution of Austin et al. (2005) and Dehnen & McLaughlin (2005)
            
        (alpha, beta=4, gamma=1): 
            the gamma-sphere studied by Dehnen (1993) and Tremaine et al. (1994) 
            
        (alpha, beta=3, gamma=1): 
            the generalized NFW profiles of Navarro, Frenk & White (1996, 1997)
            
        (alpha=1, beta, gamma=1): 
            the β-sphere of Zhao (1996) (corresponds to the alternative generalized NFW profiles studied by Evans & An 2006)
            
        (alpha=2-gamma, beta=gamma+3, gamma): 
            the hypervirial family of Evans & An (2005), introduced as the generalized isochronous model by Veltmann 1979
            
        (alpha=1-gamma/2+beta0, beta=3+gamma, gamma):  
            the phase-space power-law solutions of Dehnen & McLaughlin (2005), beta0: the anisotropy parameter at the centre

        (alpha=1/2,beta=4,gamma=0):
            Perfect sphere, finite core (de Zeeuw 1985a,b)
        
        (alpha=1/2,beta=3,gamma=0):
            Mod. Hubble profile, finite core (Binney & Tremaine 1987)
            
        (alpha=1/2,beta=2,gamma=0):
            Mod. isothermal sphere, finite core (Sackett and Sparke 1990)
        '''
        super().__init__()
        self._parameters['density_scale_radius']=density_scale_radius
        self._parameters['scale_radius']=scale_radius
        self._parameters['alpha']=alpha
        self._parameters['beta']=beta
        self._parameters['gamma']=gamma
        
        # rho(r) = rho0 * (a)^(beta) / [(r^alpha) * (a^gamma + r^gamma)^[(beta-alpha)/gamma]
        
        
        
    @classmethod
    def parameter_bounds(cls, r_values, rho_values):
        if cls.BOUND['density_scale_radius']:
            density_lower_bound = cls.BOUND['density_scale_radius'][0]
            density_upper_bound = cls.BOUND['density_scale_radius'][1]
        else:
            density_lower_bound = np.amin(rho_values)
            density_upper_bound = np.amax(rho_values)

        if cls.BOUND['scale_radius']:
            radial_lower_bound = cls.BOUND['scale_radius'][0]
            radial_upper_bound = cls.BOUND['scale_radius'][1]
        else:
            radial_lower_bound = np.amin(r_values)
            radial_upper_bound = np.amax(r_values)
        
        alpha_lower_bound = cls.BOUND['alpha'][0]
        alpha_upper_bound = cls.BOUND['alpha'][1]
        
        beta_lower_bound = cls.BOUND['beta'][0]
        beta_upper_bound = cls.BOUND['beta'][1]
        
        gamma_lower_bound = cls.BOUND['gamma'][0]
        gamma_upper_bound = cls.BOUND['gamma'][1]
        
        return ([density_lower_bound, radial_lower_bound, alpha_lower_bound, beta_lower_bound, gamma_lower_bound], 
                [density_upper_bound, radial_upper_bound, alpha_upper_bound, beta_upper_bound, gamma_upper_bound])
        
    def jacobian(self, radius):
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        alpha = self._parameters['alpha']
        beta = self._parameters['beta']
        gamma = self._parameters['gamma']
        
        #coef1 = density_scale_radius*scale_radius**beta
        coef2 = np.power(radius,alpha)
        coef3 = scale_radius**gamma+np.power(radius,gamma)
        coef4 = (beta-alpha)/gamma
        
        d_density_scale_radius = scale_radius**beta/(coef2*np.power(coef3,coef4))
        
        temp_coef1 = np.power(coef3,-coef4)
        temp_coef2 = alpha*scale_radius**(gamma+beta)+beta*np.power(radius,gamma)*scale_radius**beta
        temp_coef3 = coef2*(scale_radius**(gamma+1)+np.power(radius,gamma)*scale_radius)
        d_scale_radius = density_scale_radius*temp_coef1*temp_coef2/temp_coef3
        
        temp_coef1 = temp_coef1
        temp_coef2 = (scale_radius**beta*np.log(coef3)-gamma*scale_radius**beta*np.log(radius))
        temp_coef3 = gamma*coef2
        d_alpha = density_scale_radius*temp_coef1*temp_coef2/temp_coef3
        
        temp_coef1 = np.power(coef3,alpha/gamma)*np.log(coef3) - gamma*np.power(coef3,alpha/gamma)*np.log(scale_radius)
        temp_coef2 = gamma*coef2*np.power(coef3,beta/gamma)
        d_beta = -density_scale_radius*temp_coef1*scale_radius**beta/temp_coef2
        
        temp_coef1 = np.power(coef3,alpha/gamma)*((beta-alpha)*scale_radius**(gamma+beta)+
                        (beta-alpha)*scale_radius**beta*np.power(radius,gamma))*np.log(coef3)
        temp_coef2 = np.power(coef3,alpha/gamma)*((alpha-beta)*np.log(scale_radius)*gamma*scale_radius**(gamma+beta)
                                                  +(alpha-beta)*scale_radius**beta*np.log(radius)*gamma*np.power(radius,gamma))
        temp_coef3 = np.power(coef3,beta/gamma)*(gamma**2*np.power(radius,gamma+alpha)+coef2*gamma**2*scale_radius**gamma)
        d_gamma = density_scale_radius*(temp_coef1+temp_coef2)/temp_coef3
        
        
        return np.transpose([d_density_scale_radius, d_scale_radius,d_alpha,d_beta,d_gamma])
    
    def __call__(self,radius):
        density_scale_radius = self._parameters['density_scale_radius']
        scale_radius = self._parameters['scale_radius']
        alpha = self._parameters['alpha']
        beta = self._parameters['beta']
        gamma = self._parameters['gamma']
        coef1 = density_scale_radius*scale_radius**beta
        coef2 = np.power(radius,alpha)
        coef3 = scale_radius**gamma+np.power(radius,gamma)
        coef4 = (beta-alpha)/gamma
        
        return coef1/(coef2*np.power(coef3,coef4))
        
    
    def enclosed_mass(self, radius):
        """Return the mass, M(r), enclosed within a given radius"""
        pass
    
    def logarithmic_slope(self, radius):
        
        scale_radius = self._parameters['scale_radius']
        alpha = self._parameters['alpha']
        beta = self._parameters['beta']
        gamma = self._parameters['gamma']
        return - (alpha*scale_radius**(gamma)+beta*radius**gamma)/(scale_radius**gamma+radius**gamma)
    