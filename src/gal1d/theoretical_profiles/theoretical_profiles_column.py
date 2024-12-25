

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
        radius = np.asarray(radius)
        I_e = self._parameters['I_e']
        n = self._parameters['n']
        r_e = self._parameters['r_e']
        I = I_e * np.exp( -self.b_n_exact(n)*(np.power(radius/r_e, 1.0/n) - 1.0) )
        return I
    
    def formular(self):
        s=r'\Sigma(R) = \Sigma_e \mathrm{exp}\{- b_n [(\frac{R}{R_e})^{\frac{1}{n}}-1]\}'
        return Latex_print(s)
    
    def enclosed_mass(self, radius):
        radius = np.asarray(radius)
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
        radius = np.asarray(radius)
        h = self._parameters['h']
        I_0 = self._parameters['I_0']

        d_I_0 = np.exp(-radius/h)
        d_h = I_0*np.exp(-radius/h)*(radius/h**2)
        return np.transpose([d_I_0, d_h])
    
    def __call__(self, radius):
        radius = np.asarray(radius)
        h = self._parameters['h']
        I_0 = self._parameters['I_0']
        return I_0*np.exp(-radius/h)
    
    def formular(self):
        s=r'\Sigma(R) = \Sigma_0 \mathrm{exp}\{-\frac{R}{h}\}'
        return Latex_print(s)
    def enclosed_mass(self, radius):
        radius = np.asarray(radius)
        h = self._parameters['h']
        I_0 = self._parameters['I_0']
        coef1 = 2*np.pi*h**2*I_0
        coef2 = -2*np.pi*h*I_0*radius/np.exp(radius/h)
        coef3 = -2*np.pi*h**2*I_0/np.exp(radius/h)
        
        return coef1+coef2+coef3
    
    def logarithmic_slope(self, radius):
        radius = np.asarray(radius)
        h = self._parameters['h']
        return -1/(radius*h)
    
    def total_mass(self,ell = 0.):
        h = self._parameters['h']
        I_0 = self._parameters['I_0']
        return 2 * np.pi * I_0 * (h*h) * (1.0 - ell)
    
class CoreSersicProfile(AbstractBaseProfile):
    '''Represents a core-Sersic profile.'''
    
    
    
    BOUND = {}
    BOUND['n'] = [0,10]
    BOUND['alpha'] = [0,20]
    BOUND['gamma'] = [0,2]
    def __init__(self,I_b, r_b, r_e, n, alpha, gamma):
        '''
        Represents a core-Sersic profile.
        Graham et al. 2003, Trujillo et al. 2004
        
        Parameters
        ----------
        I_b:    float, the overall intensity scaling
        r_b:    float, the break radius, >0
        r_e:    float, the effective (half-light) radius of Sérsic profile, >0
        n:      float, the Sérsic index, >0
        alpha:  float, the smoothness of transition between the two regimes, >0
        gamma:  float, the single power law for r<r_b,      <2
        ----------
        designed to fit the profiles of so-called “core” galaxies e.g., Ferrarese et al. 2006; 
        Richings et al. 2011; Dullo & Graham 2012, 2013; Rusli et al. 2013
        '''
        super().__init__()
        self._parameters['I_b']=I_b
        self._parameters['r_b']=r_b
        self._parameters['r_e']=r_e
        self._parameters['n']=n
        self._parameters['alpha']=alpha
        self._parameters['gamma']=gamma
        
    @classmethod
    def parameter_bounds(cls, r_values, rho_values):
        I_b_lower_bound = np.amin(rho_values)
        I_b_upper_bound = np.amax(rho_values)

        r_b_lower_bound = np.amin(r_values)
        r_b_upper_bound = np.amax(r_values)
        
        r_e_lower_bound = np.amin(r_values)
        r_e_upper_bound = np.amax(r_values)
        
        n_lower_bound = cls.BOUND['n'][0]
        n_upper_bound = cls.BOUND['n'][1]
        
        alpha_lower_bound = cls.BOUND['alpha'][0]
        alpha_upper_bound = cls.BOUND['alpha'][1]
        
        gamma_lower_bound = cls.BOUND['gamma'][0]
        gamma_upper_bound = cls.BOUND['gamma'][1]

        return ([I_b_lower_bound, r_b_lower_bound, r_e_lower_bound, n_lower_bound, alpha_lower_bound, gamma_lower_bound], 
                [I_b_upper_bound, r_b_upper_bound, r_e_upper_bound, n_upper_bound, alpha_upper_bound, gamma_upper_bound])
        
        
    def jacobian(self, radius):
        radius = np.asarray(radius)
        I_b = self._parameters['I_b']
        r_b = self._parameters['r_b']
        r_e = self._parameters['r_e']
        n = self._parameters['n']
        alpha = self._parameters['alpha']
        gamma = self._parameters['gamma']
        bn = SersicProfile.b_n_exact(n)
        
        coef1 = np.power(np.power(r_b/radius,alpha)+1,gamma/alpha)
        coef2 = (np.power(radius,alpha)+r_b**alpha)/r_e**alpha
        coef3 = np.exp(-bn*np.power(coef2,1/(n*alpha)))
        d_I_b = coef1*coef3
        
        temp_coef1 = I_b*coef1
        temp_coef2 = bn*r_b**alpha*np.power(coef2,1/(n*alpha))-n*gamma*r_b**alpha
        temp_coef3 = coef3
        temp_coef4 = n*(r_b**(alpha+1)+np.power(radius,alpha)*r_b)
        d_r_b = -temp_coef1*temp_coef2*temp_coef3/temp_coef4
        
        temp_coef1 = bn*I_b*coef1
        temp_coef2 =  np.power(coef2,1/(n*alpha))
        temp_coef3 = coef3
        d_r_e = temp_coef1*temp_coef2*temp_coef3/(n*r_e)
        
        # use bn = 1.9992n − 0.3271 for 0.5 <n< 10 to cal d(bn)/dn 
        #  Capaccioli (1989) (see also Prugniel & Simien 1997, their equation A3a
        temp_coef1 = -2 * coef1 * I_b
        temp_coef2 = np.power(coef2,1/(n*alpha))*(1 - np.log(coef2)/(alpha*n))
        temp_coef3 = coef3
        d_n = temp_coef1*temp_coef2*temp_coef3
        
        temp_coef1 = gamma*((np.power(r_b/radius,alpha)*np.log(r_b/radius)*alpha/(np.power(r_b/radius,alpha)+1))
                            -np.log(np.power(r_b/radius,alpha)+1))
        temp_coef2 = np.exp((gamma*np.log(np.power(r_b/radius,alpha)+1)/alpha)-bn*np.power(coef2,1/(n*alpha)))
        temp_coef3 = temp_coef1*temp_coef2/alpha**2
        temp_coef4 = bn*coef1* np.exp(np.log(coef2/alpha/n)-bn*np.power(coef2,1/(n*alpha)))
        temp_coef5 = alpha*(r_e**alpha*(np.log(radius)*np.power(radius,alpha)+np.log(r_b)*r_b**alpha)
                            -np.log(r_e)*r_e**alpha*(np.power(radius,alpha)+r_b**alpha))/(r_e**alpha*(np.power(radius,alpha)+r_b**alpha))
        temp_coef6 = temp_coef4*(temp_coef5-np.log(coef2))/(n*alpha**2)
        d_alpha = I_b*(temp_coef3-temp_coef6)
        
        d_gamma = I_b*coef1*coef3*np.log(np.power(r_b/radius,alpha)+1)/alpha
        
        return np.transpose([d_I_b, d_r_b, d_r_e, d_n, d_alpha, d_gamma])
    
    def __call__(self, radius):
        radius = np.asarray(radius)
        I_b = self._parameters['I_b']
        r_b = self._parameters['r_b']
        r_e = self._parameters['r_e']
        n = self._parameters['n']
        alpha = self._parameters['alpha']
        gamma = self._parameters['gamma']
        bn = SersicProfile.b_n_exact(n)
        
        coef1 = I_b*np.power(np.power(r_b/radius,alpha)+1,gamma/alpha)
        coef2 = (np.power(radius,alpha)+r_b**alpha)/r_e**alpha
        coef3 = np.exp(-bn*np.power(coef2,1/(n*alpha)))
        return coef1*coef3
    
    
    def formular(self):
        s=r'\Sigma(R) = \Sigma_b [1+(\frac{R_b}{R})^{\alpha}]^{\gamma/\alpha}\mathrm{exp}\{-b (\frac{R^{\alpha}+R_b^{\alpha}}{R_e^{\alpha}})^{1/(n\alpha)}\}'
        return Latex_print(s) 
    def logarithmic_slope(self,radius):
        radius = np.asarray(radius)
        
        #I_b = self._parameters['I_b']
        r_b = self._parameters['r_b']
        r_e = self._parameters['r_e']
        n = self._parameters['n']
        alpha = self._parameters['alpha']
        gamma = self._parameters['gamma']
        bn = SersicProfile.b_n_exact(n)
        coef1 = bn*np.power(radius, alpha)*np.power((np.power(radius,alpha)+r_b**alpha)/r_e**alpha,1/(n*alpha))+n*r_b**alpha*gamma
        coef2 = n*(np.power(radius,alpha)+r_b**alpha)
        return - coef1/coef2
        
        
        pass
    def enclosed_mass(self, radius):
        pass