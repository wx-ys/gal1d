

'''

added profiles: SersicProfile, ExponentialProfile
'''

from .base_utils import *



class SersicProfile(AbstractBaseProfile):
    '''Represents a Sersic profile.'''
    
    BOUND = {}
    BOUND['n'] = [0.2,8]
    BOUND['r_e'] = None
    BOUND['I_e'] = None
    
    def __init__(self,n,r_e,I_e):
        '''Represents a Sersic profile. Sérsic’s (1963, 1968) R^(1/n)
        
        Parameters
        ----------
        n:      float, the Sérsic index controlling the shape of the intensity profile
        r_e:    float, the effective (half-light) radius r_e
        I_e:    float, the surface brightness at the effective (half-light) radius r_e
        ----------
        equivalent to the de Vaucouleurs “r 1/4 ” profile when n = 4, 
        to an exponential when n = 1, 
        and to a Gaussian when n = 0.5.
        '''
        super().__init__()
        self._parameters['n']=n
        self._parameters['r_e']=r_e
        self._parameters['I_e']=I_e
        
    @classmethod
    def parameter_bounds(cls, r_values, rho_values):
        if cls.BOUND['r_e']:
            r_e_lower_bound = cls.BOUND['r_e'][0]
            r_e_upper_bound = cls.BOUND['r_e'][1]
        else:
            r_e_lower_bound = np.amin(r_values)
            r_e_upper_bound = np.amax(r_values)
            
        if cls.BOUND['I_e']:
            I_e_lower_bound = cls.BOUND['I_e'][0]
            I_e_upper_bound = cls.BOUND['I_e'][1]
        else:
            I_e_lower_bound = np.amin(rho_values)
            I_e_upper_bound = np.amax(rho_values)
        
        
        n_lower_bound = cls.BOUND['n'][0]
        n_upper_bound = cls.BOUND['n'][1]
        
        return ([n_lower_bound, r_e_lower_bound, I_e_lower_bound], 
                [n_upper_bound, r_e_upper_bound, I_e_upper_bound])
        
    def jacobian(self,radius):
        I_e = self._parameters['I_e']
        n = self._parameters['n']
        r_e = self._parameters['r_e']
        d_I_e = np.exp( -self.b_n_exact(n)*(np.power(radius/r_e, 1.0/n) - 1.0) )
        
        d_r_e = (I_e * np.exp( -self.b_n_exact(n)*(np.power(radius/r_e, 1.0/n) - 1.0))
                 *(-self.b_n_exact(n))*(1/n)*np.power(radius/r_e, 1.0/n-1)*(-radius/r_e**2) )
        # use bn = 1.9992n − 0.3271 for 0.5 <n< 10 to cal d(bn)/dn 
        #  Capaccioli (1989) (see also Prugniel & Simien 1997, their equation A3a
        d_n = (I_e * np.exp( -self.b_n_exact(n)*(np.power(radius/r_e, 1.0/n) - 1.0))
                 *(-self.b_n_exact(n))*np.power(radius/r_e, 1.0/n)*np.log(radius/r_e)*(1/n**2)
                 *(-1.99992*(np.power(radius/r_e, 1.0/n))))  # use bn = 1.9992n − 0.3271 to cal d(bn)/dn # Capaccioli (1989)
        return np.transpose([d_n, d_r_e, d_I_e])

        
    @staticmethod
    def b_n_exact(n):
        """Exact calculation of the Sersic derived parameter b_n, via solution
        of the function
                Gamma(2n) = 2 gamma_inc(2n, b_n)
        where Gamma = Gamma function and gamma_inc = lower incomplete gamma function.

        If n is a list or Numpy array, the return value is a 1-d Numpy array
        from pyimfit
        b n is calculated via the polynomial approximation of Ciotti & Bertin [1999] when n > 0.36 
        and the approximation of MacArthur, Courteau, & Holtzman [2003] when n ≤ 0.36
        """
        def myfunc(bn, n):
            return abs(float(2*GammaInc(2*n, 0, bn) - Gamma(2*n)))
        if np.iterable(n):
            b = [scipy.optimize.brent(myfunc, (nn,)) for nn in n]
            b = np.array(b)
        else:
            b = scipy.optimize.brent(myfunc, (n,))
        return b
    
    def __call__(self,radius):
        I_e = self._parameters['I_e']
        n = self._parameters['n']
        r_e = self._parameters['r_e']
        I = I_e * np.exp( -self.b_n_exact(n)*(np.power(radius/r_e, 1.0/n) - 1.0) )
        return I
    
    def enclosed_mass(self, radius):
        I_e = self._parameters['I_e']
        n = self._parameters['n']
        r_e = self._parameters['r_e']
        
        bn = self.b_n_exact(n)
        x = bn*np.power((radius/r_e),1/n)  # Graham et al. 1996, equation 8
        bn2n = bn**(2*n)
        
        coef1 = I_e*r_e**2*2*np.pi
        coef2 = n*np.exp(bn)/bn2n
        coef3 = GammaInc(2*n,0,x)
        L = coef1*coef2*coef3

        return L
    
    def total_mass(self, ell = 0.):
        I_e = self._parameters['I_e']
        n = self._parameters['n']
        r_e = self._parameters['r_e']
        
        bn = self.b_n_exact(n)
        bn2n = bn**(2*n)
        totalmass = 2 * np.pi * n * np.exp(bn) * I_e * (r_e*r_e) * (1.0 - ell)
        totalmass = totalmass * Gamma(2*n) / bn2n
        return totalmass
    
    def logarithmic_slope(self, radius):
        ''' logarithmic slope  used by Gunn & Oke (1975) and Hoessel (1980)'''
        I_e = self._parameters['I_e']
        n = self._parameters['n']
        r_e = self._parameters['r_e']
        bn = self.b_n_exact(n)
        x = bn*np.power((radius/r_e),1/n)  # Graham et al. 1996, equation 8
        coef1 = np.exp(-x)*np.power(x,2*n)
        coef2 = n*GammaInc(2*n,0,x)
        
        return coef1/coef2


class ExponentialProfile(AbstractBaseProfile):
    '''Represents an exponential profile.'''
    
    def __init__(self, I_0, h):
        '''Represents an exponential profile. 
        
        Parameters
        ----------
        I_0:    float, the central surface brightness (density)
        h:    float, the scale length
        ----------
        '''
        super().__init__()
        self._parameters['h']=h
        self._parameters['I_0']=I_0
    
    @classmethod
    def parameter_bounds(cls, r_values, rho_values):
        I_0_lower_bound = np.amin(rho_values)
        I_0_upper_bound = np.amax(rho_values)

        h_lower_bound = np.amin(r_values)
        h_upper_bound = np.amax(r_values)

        return ([I_0_lower_bound, h_lower_bound], [I_0_upper_bound, h_upper_bound])
    
    def jacobian(self, radius):
        h = self._parameters['h']
        I_0 = self._parameters['I_0']

        d_I_0 = np.exp(-radius/h)
        d_h = I_0*np.exp(-radius/h)*(radius/h**2)
        return np.transpose([d_I_0, d_h])
    
    def __call__(self, radius):
        h = self._parameters['h']
        I_0 = self._parameters['I_0']
        return I_0*np.exp(-radius/h)
    
    def enclosed_mass(self, radius):
        h = self._parameters['h']
        I_0 = self._parameters['I_0']
        coef1 = 2*np.pi*h**2*I_0
        coef2 = -2*np.pi*h*I_0*radius/np.exp(radius/h)
        coef3 = -2*np.pi*h**2*I_0/np.exp(radius/h)
        
        return coef1+coef2+coef3
    
    def logarithmic_slope(self, radius):
        h = self._parameters['h']
        return -1/(radius*h)
    
    def total_mass(self,ell = 0.):
        h = self._parameters['h']
        I_0 = self._parameters['I_0']
        return 2 * np.pi * I_0 * (h*h) * (1.0 - ell)