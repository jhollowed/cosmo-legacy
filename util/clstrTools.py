'''
Joe Hollowed
Last edited 5/20/2017

Providing set of tools for quick calculations and conversions of cluster physics
'''

import pdb
import math
import numba
import numpy as np
import simTools as st
import massConversion as mc
from astropy import units as u
from astropy.cosmology import WMAP7
from astropy import constants as const
from astropy.coordinates import SkyCoord
from halotools.empirical_models import NFWProfile

c = const.c.value
Msun = const.M_sun.value


# ==========================================================================================


def LOS_properVelocity(zi,z, dzi = [], dz = None):
    '''
    Returns the line of sight peculiar velocity of a galaxy or galaxies (Ruel et al. 2014).

    :param zi: an array-like of redshift values
    :param z: the redshift of the cluster that contains the member galaxy
              (biweight average of z)
    :param dzi: error in redshift values
    :param dz: error in cluster redshift
    :returns: array of proper velocities in km/s
    '''
    zi = np.array(zi) 
    num = c*(zi-z)
    den = (1+z)
    v = (num / den) / 1000

    if(len(dzi) == 0 and dz == None):
        return v
    else:
        dzi = np.array(dzi)
        err_num = c*np.sqrt(dzi**2 + dz**2)
        errRel_num = err_num / num
        err_den = dz
        errRel_den = dz/z
        dv = np.sqrt( errRel_num**2 + errRel_den**2) / 1000 * v
        return [v, dv]


# ==========================================================================================


def projectedDist(coords, center_coords, z, cosmo = WMAP7, dist_type='comoving'):
    '''
    Convert RA and Dec coordinate values to angular separation, and then separation distance 
    in Mpc
    :param coords: Array of galaxy coordinates tuples or lists in the form (ra, dec)
    :param center_coords: The coordinate of the cluster center in the form (ra, dec)
    :param z: redshift of cluster
    :param cosmo: AstroPy Cosmology object (default is cosmo = WMAP7)
    :param dist_type: whether to return proper or comoving distance measurements 
                      (default is type=comoving)
    :return: numpy array of projected distance values (if units = 0) or AstroPy Quantity 
	     objects (if units = 1) in units of Mpc (if type=proper) or Mpc/h (if type=comoving)
    '''

    coords = np.array(coords)
    dist = np.zeros(len(coords), dtype=object)
    center = SkyCoord(ra=center_coords[0], dec=center_coords[1], unit='deg')
    h = cosmo.h
    
    if (dist_type == 'comoving'): kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).value
    elif (dist_type == 'proper'): kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).value
    else: raise ValueError('type must be either \'proper\' or \'comoving\'')
    
    skyCoords = SkyCoord(ra=coords.T[0], dec=coords.T[1], unit='deg')
    sep = skyCoords.separation(center).to('arcmin').value
    dist = sep * kpc_per_arcmin / 1000
    return dist


# ===========================================================================================


def interpolated_kpc_per_arcmin(z, cosmo = WMAP7, dist_type = 'comoving'):
    '''
    Interpolate the astropy kpc_per_arcmin functions
    :param z: array of redshifts at which to preform the interpolation
    :param cosmo: an astropy cosmology object instance (default = WMAP7)
    :param dist_type: whether to return comoving or proper distances
    :return: the kpc per arcmin separation at the specified redshifts (input param z)
    '''
    
    try: samplePoints = np.load('interpolated_kpc_{}_per_arcmin.npy'.format(dist_type))
    except FileNotFOundError: 
        zspace = np.linspace(0, 2, 2000)
        if(dist_type == 'comoving'): kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).value
        elif(dist_type == 'proper'): kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z).value
        else: raise ValueError('type must be either \'proper\' or \'comoving\'')
        cols = ['z', 'kpc']
        samplePoints = np.rec.fromarrays([zspace, kpc_per_arcmin], names=cols)
        np.save('interpolated_kpc_{}_per_arcmin.npy'.format(dist_type))
    z_pts = samplePoints['z']
    kpc_pts = samplePoints['kpc']
    kpc_interp = np.interp(z, z_pts, kpc_pts)
    pdb.set_trace()
    return kpc_interp
    

# ===========================================================================================


def mass_to_radius(mass, z, h = 100, mdef ='200c', cosmo = WMAP7, Msun=1):
    '''
    spherical radius overdensity as a function of the input mass usign halotools
    :param mass: cluster mass(es) in units of M_sun/h or M_sun/h70 
                 (factor of 10^14 is assumed)
    :param z: redshift of given cluster
    :param h: the kind of dimensionless hubble parameter (either h (h=100) or h70 (h=70), 
	          default is h = 100)
    :param mdef: mass definition (default is mdef = '200c')
    :param cosmo: AstroPy Cosmology object instance (default is cosmo = WMAP7)
    :param Msun: whether ot not masses are given in units of M_sun. If 0, then divide all 
                 masses by M_sun. If 1, don't do anything (default is Msun=1)
    :return: scalar or array of halo radii in Mpc/h with the same overdesnity as the mass 
             definition
    '''

    mass = np.atleast_1d(mass)
    z = np.atleast_1d(z)
    N = len(mass)

    if(Msun == 0): mass = mass / const.M_sun.value

    if(h == 70):
        mass = (mass / st.h70()) * cosmo.h
    elif(h != 100):
        raise ValueError('Masses must be given in units of M_sun/h70 or M_sun/h')
        return None

    radii = np.zeros(N)

    for i in range(N):
        profile = NFWProfile(cosmology=cosmo, redshift=z[i], mdef=mdef)
        radii[i] = profile.halo_mass_to_halo_radius(mass[i])

    if(len(radii) > 1): return radii
    else: return radii[0]


# ===========================================================================================


def richness_to_m200(l):
    '''
    Courtesy of Dan
    mass richness relation from http://arxiv.org/abs/1603.06953
    
    :param l: richness of cluster
    :return: m200, the mass of the cluster with respect to the critical density, 
                   in units of M_sun/h_70
    '''
    m200 = 1e14*np.exp(1.48 + 1.06*np.log(l/60.0))
    return m200


# ==========================================================================================


def richness_to_arcmin(l,z, cosmo = WMAP7, mdef='200c', h=70):
    mass = richness_to_m200(l)
    radius = mass_to_radius(mass, z, h, mdef =mdef)
    # multiply radius by h and divide denominator by 1000
    # since halotools halo_mass_to_halo_radius() returns radius in units of Mpc/h
    arcmin = (radius / cosmo.h) / (cosmo.kpc_proper_per_arcmin(z).value / 1000)
    return arcmin


# ===========================================================================================


def classifyType(spectra):
    '''
    Classifies galaxies by type according to criteria given by Bayliss et al. (2016) =, 
    Table 5.
    :param spectra: list of lists. Each list is a galaxies spectral info including their 
                    OII emission, H_delta emission, and certainties for each expressed in 
                    sigmas. 
		            Expected form: [OII, OII_erorr, Hd, Hd_error]
    :return: list of galaxy types in order passed. 0 = passive, 
                                                   1 = post-starburst, 
                                                   2 = star-forming
    '''

    types = np.zeros(len(spectra), dtype = int)

    for n in range(len(spectra)):
        OII, sigOII, Hd, sigHd = spectra[n]
        if(abs(OII) > 998 or sigOII <= 2.0):
            if(Hd < 3): types[n] = 0
            elif(Hd >= 3): types[n] = 1
        else: types[n] = 2

    return types


# ==========================================================================================


def convertMass(mass, mdef, z, mdef_out = 200, h = 100, cosmo = WMAP7):
    '''
    Convert between mass definitions according to NFW profile
    :param mass: mass in some definition (array of scalar)
    :param mdef: input mass definition (usually 500 or 200)
    :param mdef_out: the output mass definition, to be used if mdef=200
    :param z: redshift of cluster
    :param cosmo: AstroPy cosmology instance
    :param h: h used to define input mass (typically 100 or 70)
    :return: mass in output unit
    '''

    mass = np.atleast_1d(mass)
    z = np.atleast_1d(z)

    if (h == 70):
        mass = (mass / st.h70()) * cosmo.h
    elif (h != 100):
        raise ValueError('Masses must be given in units of M_sun/h70 or M_sun/h')
        return

    newMass = np.zeros(len(mass))

    if(mdef == 200):
        for i in range(len(mass)):
            newMass[i] = mc.M200_to_MDelta(mass[i], mdef_out, z[i])
    else:
        if(mdef_out != 200): 
            raise ValueError('if (mdef != 200), then it must be that (mdef_out==200)')
        for i in range(len(mass)):
            newMass[i] = mc.MDelta_to_M200(mass[i], mdef, z[i])

    return newMass


# ===========================================================================================


def meanVirialDensity(z, cosmo = WMAP7):
    '''
    Compute the virial overdensity of a halo with respect to the
    mean matter density (Hu&Kravtsov 2008)
    :param z: redshift of cluster
    :param cosmo: AstroPy cosmology instance (default is cosmo=WMAP7)
    :return: delta_v, the virial overdensity
    '''
    x = cosmo.Om(z) - 1
    dv_n = (18*(np.pi**2)) + (82*x) - (39*(x**2))
    dv_d = 1 + x
    dv = dv_n/dv_d
    return dv
