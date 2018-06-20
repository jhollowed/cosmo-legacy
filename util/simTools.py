'''
Joe Hollowed
COSMO-HEP 2017

Providing a set of tools for quick cosmological calculations
'''
import pdb
import numpy as np
from astropy.cosmology import WMAP7


# ==================================================================================================


def h70(z = 0, cosmo = WMAP7):
    '''
    Return the value of h_70 at a redshift z
    :param z: redshift at which to measure the Hubble parameter (default is z=0)
    :param cosmo: instance of an astropy Cosmology object (default is WMAP7)
    :return: h_70
    '''
    h70 = cosmo.H(z).value / 70
    return h70


def h(z = 0, cosmo = WMAP7):
    '''
    Return the value of h at a redshift z
    :param z: redshift (or array of redshifts) at which to measure the Hubble parameter (default is z=0)
    :param cosmo: instance of an astropy Cosmology object
    :return: h, the hubble constant over 100
    '''
    h = cosmo.H(z).value / 100
    return h


# ==================================================================================================


class StepZ:
    '''
    Tool for converting between simulation step number and redshift/scale factor
    From Dan's Toolkit
    '''

    def __init__(self,start_z,end_z,num_steps):
        self.z_in = start_z
        self.z_out = end_z
        self.num_steps = float(num_steps)
        self.a_in = 1./(1.+start_z)
        self.a_out = 1./(1.+end_z)
        self.a_del = (self.a_out- self.a_in)/(self.num_steps-1)
    
    def get_z(self,step):
        #to get rid of annoying rounding errors on z=0 (or other ending values)
        #if(step == self.num_steps-1):
        #    return 1./self.a_out -1. 
        a = self.a_in+step*self.a_del
        return 1./a-1.
    
    def get_step(self,z):
        a = 1./(z+1.)
        return np.ceil( (a-self.a_in)/self.a_del).astype(int)

    def get_a(self,step):
        return 1./(self.get_z(step)+1.)


# ==================================================================================================


def evrardRelation(sigma = None, m200 = None, z = 0, sigDM15 = 1082.9, sigDM15_err = 4.0, 
                   a = 0.3361, a_err = 0.0026):
    '''
    Returns the inferred dark matter (subhalo) velocty dispersion with a
    measured SZ-based mass, via the relation provided by Evrard et al.
    :param sigma: The velocity dispersion(s) of the cluster(s)
    :param mass: cluster mass(es) expressed in M_200
    :param z: cluster redshift(s)
    :param sigDM15: log slope parameter (default best fit from Evrard+ 2003)
    :param sigDM15_err: error in fit parameter sigDM15 (default best fit from Evrard+ 2003)
    :param a: log intercept parameter (best fit from Evrard+ 2003)
    :param a_err: error in fit parameter a (default best fit from Evrard+ 2003)
    :return: if m200 passed, return inferred velocity dispersion (in km s^-1).
             if sigma passed, return inferred mass (in m200)
    '''

    if(sum([val is None for val in [sigma, m200]]) != 1):
        raise ValueError('Either the sigma or mass must be passed (not none, and not both)')
    
    if(sigma == None):

        # calculate velocty dispersion using input mass
        sigma = sigDM15 * ( (h(z)*m200) / 1e15 )**a

        # error propegation
        partial_a = sigDM15 * a * (h(z)*m200/1e15)**(a-1)
        partial_sig = (h(z)*m200/1e15) ** a
        sigma_err = np.sqrt( (partial_a * a_err)**2 + (partial_sig * sigDM15_err)**2)
        
        return np.array([sigma, sigma_err])

    elif(m200 == None):

        # calculate mass using inpit velocity dispersion
        m200 = 1e15 * (sigma/sigDM15)**(1/a) / h(z)

        # error propegation
        partial_a = - ((1e15/h(z)) * (sigma/sigDM15)**(1/a) * np.log(sigma/sigDM15)) / a**2
        partial_sig = - ((1e15/h(z)) * (sigma/sigDM15)**(1/a)) / (a*sigDM15)
        m200_err =  np.sqrt( (partial_a * a_err)**2 + (partial_sig * sigDM15_err)**2)

        return np.array([m200, m200_err])
