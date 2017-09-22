'''
Joe Hollowed
COSMO-HEP 2017

Providing a set of tools for quick cosmological calculations
'''

import numpy as np
from astropy.cosmology import WMAP7 as cosmo

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
        return (a-self.a_in)/self.a_del

    def get_a(self,step):
        return 1./(self.get_z(step)+1.)

