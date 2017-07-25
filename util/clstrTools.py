'''
Joe Hollowed
Last edited 5/20/2017

Providing set of tools for quick calculations and conversions of cluster physics
'''

from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9
from halotools.empirical_models import NFWProfile
from astropy import units as u
import numpy as np
import pdb
import massConversion as mc

c = const.c.value
Msun = const.M_sun.value




def LOS_properVelocity(zi,z, dzi = [], dz = None):
    '''
    Returns the line of sight proper velocity of a galaxy or galaxies (Ruel et al. 2014).

    :param zi: an array-like of redshift values
    :param z: the redshift of the cluster that contains the member galaxy (biweight average of z)
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


def projectedDist(coords, center_coords, z, cosmo = WMAP9, type='comoving', units=0):
    '''
    Convert RA and Dec coordinate values to angular separation, and then separation distance in Mpc
    :param coords: Array of galaxy coordinates tuples or lists in the form (ra, dec)
    :param center_coords: The coordinate of the cluster center in the form (ra, dec)
    :param z: redshift of cluster
    :param cosmo: AstroPy Cosmology object (default is cosmo = WMAP9)
    :param type: whether to return proper or comoving distance measurements (default is type=comoving)
    :param units: boolean argument to return distances as AstroPy Quantity objects or not 
		  (default is units=0)
    :return: numpy array of projected distance values (if units = 0) or AstroPy Quantity 
	     objects (if units = 1) in units of Mpc (if type=proper) or Mpc/h (if type=comoving)
    '''

    coords = np.array(coords)
    sep = np.zeros(len(coords), dtype=object)
    dist = np.zeros(len(coords), dtype=object)
    center = SkyCoord(ra=center_coords[0], dec=center_coords[1], unit='deg')
    h = cosmo.h

    for n in range(len(coords)):
        nextCoord = coords[n]
        coord = SkyCoord(ra=nextCoord[0], dec=nextCoord[1], unit='deg')
        sep[n] = coord.separation(center).to('arcmin')
        if(type=='proper'): 
            dist[n] = sep[n] * h * (cosmo.kpc_proper_per_arcmin(z)/1000) * (u.Mpc/u.kpc)
        if(type=='comoving'): 
            dist[n] = sep[n] * h * (cosmo.kpc_comoving_per_arcmin(z)/1000) * (u.Mpc/u.kpc)

    if(units == 0): return np.array([float(d.value) for d in dist])
    else: return np.array(dist)



def mass_to_radius(mass, z, h = 100, mdef ='200c', cosmo = WMAP9, Msun=1):
    '''
    spherical radius overdensity as a function of the input mass usign halotools
    :param mass: cluster mass(es) in units of M_sun/h or M_sun/h70 (factor of 10^14 is assumed)
    :param z: redshift of given cluster
    :param h: the kind of dimensionless hubble parameter (either h (h=100) or h70 (h=70), 
	      default is h = 100)
    :param mdef: mass definition (default is mdef = '200c')
    :param cosmo: AstroPy Cosmology object instance (default is cosmo = WMAP9)
    :param Msun: whether ot not masses are given in units of M_sun. If 0, then divide all masses
                 by M_sun. If 1, don't do anything (default is Msun=1)
    :return: scalar or array of halo radii in Mpc/h with the same overdesnity as the mass definition
    '''

    mass = np.atleast_1d(mass)
    z = np.atleast_1d(z)
    N = len(mass)

    if(Msun == 0): mass = mass / const.M_sun.value

    if(h == 70):
        mass = (mass / h70()) * cosmo.h
    elif(h != 100):
        raise ValueError('Masses must be given in units of M_sun/h70 or M_sun/h')
        return None

    radii = np.zeros(N)

    for i in range(N):
        profile = NFWProfile(cosmology=cosmo, redshift=z[i], mdef=mdef)
        radii[i] = profile.halo_mass_to_halo_radius(mass[i])

    if(len(radii) > 1): return radii
    else: return radii[0]



def richness_to_m200(l):
    '''
    Courtesy of Dan
    mass richness relation from http://arxiv.org/abs/1603.06953
    
    :param l: richness of cluster
    :return: m200, the mass of the cluster with respect to the critical density, in units of M_sun/h_70
    '''
    m200 = 1e14*np.exp(1.48 + 1.06*np.log(l/60.0))
    return m200



def richness_to_arcmin(l,z, cosmo = WMAP9, mdef='200c', h=70):
    mass = richness_to_m200(l)
    radius = mass_to_radius(mass, z, h, mdef =mdef)
    # multiply radius by h and divide denominator by 1000
    # since halotools halo_mass_to_halo_radius() returns radius in units of Mpc/h
    arcmin = (radius / cosmo.h) / (cosmo.kpc_proper_per_arcmin(z).value / 1000)
    return arcmin



def classifyType(spectra):
    '''
    Classifies galaxies by type according to criteria given by Bayliss et al. (2016) =, Table 5.
    :param spectra: list of lists. Each list is a galaxies spectral info including their OII emission, 
		    H_delta emission, and certainties for each expressed in sigmas. 
		    Expected form: [OII, OII_erorr, Hd, Hd_error]
    :return: list of galaxy types in order passed. 0 = passive, 1 = post-starburst, 2 = star-forming
    '''

    types = np.zeros(len(spectra), dtype = int)

    for n in range(len(spectra)):
        OII, sigOII, Hd, sigHd = spectra[n]
        if(abs(OII) > 998 or sigOII <= 2.0):
            if(Hd < 3): types[n] = 0
            elif(Hd >= 3): types[n] = 1
        else: types[n] = 2

    return types


def convertMass(mass, mdef, z, mdef_out = 200, h = 100, cosmo = WMAP9):
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
        mass = (mass / h70()) * cosmo.h
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


def meanVirialDensity(z, cosmo = WMAP9):
    '''
    Compute the virial overdensity of a halo with respect to the
    mean matter density (Hu&Kravtsov 2008)
    :param z: redshift of cluster
    :param cosmo: AstroPy cosmology instance (default is cosmo=WMAP9)
    :return: delta_v, the virial overdensity
    '''
    x = cosmo.Om(z) - 1
    dv_n = (18*(np.pi**2)) + (82*x) - (39*(x**2))
    dv_d = 1 + x
    dv = dv_n/dv_d
    return dv




def h70(z = 0, cosmo = WMAP9):
    '''
    Return the value of h_70 at a redshift z
    :param z: redshift at which to measure the Hubble parameter (default is z=0)
    :param cosmo: instance of an astropy Cosmology object (default is WMAP9)
    :return: h_70
    '''
    h70 = (cosmo.H(z) / (70 * u.km/(u.Mpc *u.s))).value
    return h70


def h(z = 0, cosmo = WMAP9):
    '''
    Return the value of h_70 at a redshift z
    :param z: redshift at which to measure the Hubble parameter (default is z=0)
    :param cosmo: instance of an astropy Cosmology object
    :return: h_70
    '''
    h = (cosmo.H(z) / (100 * u.km/(u.Mpc *u.s))).value
    return h


def saroRelation(sigBI, z, A=939, B=2.91, C=0.33):
    '''
    Returns the inferred mass of a cluster with a measured
    velocity dispersion, via the realtion provided by Saro et al. (2013)
    :param sigBI: the measured velocity dispersion of the cluster
    :param z: The redshift of the cluster
    :param A: constant
    :param B: constant
    :param C: constant
    :return: inferred mass from given velocity dispersion in units of Msun
    '''
    hz = np.array([h70(zi) for zi in z])
    mass = ((sigBI / (A*(hz**C)))**B) * 1e15
    return mass



def evrardRelation(mass, z, sigDM15 = 1082.9, a = 0.3361):
    '''
    Returns the inferred dark matter (subhalo) velocty dispersion with a
    measured SZ-based mass, via the relation provided by Evrard et al.
    :param mass: SZ-based cluster mass(es) expressed in Solar masses
    :param z: cluster redshift(s)
    :param sigDM15: constant
    :param a: constant
    :return: inferred velocity dispersion from given SZ-based masses (in km s^-1)
    '''

    hz = np.array([h(zi) for zi in z])
    sigDM = sigDM15 * (((hz*mass)/1e15)**a)
    return sigDM
