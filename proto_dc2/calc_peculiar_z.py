'''
Joe Hollowed
COSMO-HEP 2017
'''

import pdb
import numpy as np
import astropy.constants as const
from numpy.core import umath_tests as npm
from astropy.cosmology import WMAP7 as cosmo


def pecZ(x, y, z, vx, vy, vz, z_hubb=0, obs=np.zeros(3), vPec_only=False):
    '''
    This function calculates peculiar zs for n-body simulation objects, given
    their comoving position and velocity, and returns some other useful products of the calculation. 

    :param x: x-position for each object, in form [x1, x2,... xn]
    :param y: y-position for each object, in form of param x
    :param z: z-position for each object, in form of param x
    :param vx: x-velocity for each object, in form [vx1, vx2,... vxn]
    :param vy: y-velocity for each object, in form of param vx
    :param vz: z-velocity for each object, in form of param vx
    :param z_hubb: cosmological redshifts for each object, in form [z1, z2,... zn]
    :param obs: The coordinates of the observer, in form [x, y, z]
    :param vPec_only: end function early and return just the comoving peculiar velocity
                      (this is probably the only case in which you'd want to leave the 
                       z_hubb parameter at it's default of 0)
    :return: - the peculair z_hubb in each object in form of param redshift
             - the total observed z_hubb (cosmological+peculiar)
             - the peculiar velocity of each object, in the form of param vx, where negative 
               velocities are toward the observer, in comoving km/s
             - the line-of-sight velocity, in proper km/s (peculiar velocity * a)
             - distance from observer to object in comoving Mpc
             - distance from observer to object in kpc proper (comoving dist * a)
             - distorted distance from observer to object in comoving Mpc 
             - distorted distance from observer to object in proper Mpc 
    '''
    
    # get relative position (r vector) of and position unit vector toward each object
    r_rel = np.array([x, y, z]).T - obs
    r_rel_mag = np.linalg.norm(r_rel, axis=1)
    r_rel_hat = np.divide(r_rel, np.array([r_rel_mag]).T)
    
    # dot velocity vectors with relative position unit vector to get peculiar velocity
    v = np.array([vx, vy, vz]).T
    v_mag = np.linalg.norm(v, axis=1)
    v_pec = npm.inner1d(v, r_rel_hat)
    if(vPec_only): return v_pec

    # find total and peculiar z_hubb (full relativistic expression)
    c = const.c.value / 1000
    z_pec = np.sqrt( (1+v_pec/c) / (1-v_pec/c)) - 1
    z_obs = (1+z_hubb)*(1+z_pec) - 1

    # find the distorted distance from appliying Hubble's law using the new z_obs z_hubbs
    a = 1/(1+z_hubb)
    r_dist = r_rel_mag + v_pec/100./cosmo.efunc(z_hubb)/a 
    r_err = v_pec * (1+z_hubb) / (cosmo.H(z_hubb).value)
    r_distorted = r_err + r_rel_mag
    pdb.set_trace()

    return z_pec, z_obs, v_pec, v_pec*a, r_rel_mag, r_rel_mag*a, r_distorted, r_distorted*a
