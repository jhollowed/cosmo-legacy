'''
Joe Hollowed
Last edited 2016

This module provides several functions for removing interlopers from galaxy cluster datasets.
I've written these and compiled them in this module to allow easy switching between member
selection methods which may each provide different benefits.
'''

import dispersionStats as stat
from clstrTools import LOS_properVelocity, projectedDist
import numpy as np
import astropy.constants as const
from astropy.cosmology import WMAP9
import pdb

def sig_interlopers(galZs, galZs_err=[], vCut=4500, binZ=True):
    '''
    Preform 3-sigma clipping on a set of galaxy redshifts, by computing their corresponding proper
    velocity. Proper velocity is found by finding the biweight average in the galaxy redshifts and
    considering it to be the cluster redshift. Once velocities are found, an initial hard velocity
    cut is applied, then clipping begins until it converges.
    :param galZs: array-like of galaxy redshifts
    :param galZs_err: error in galaxy redshifts (optional)
    :param vCut: velocity in km/s to cut at before clipping
    :return: boolean clip mask, galaxy velocities (in km/s), and cluster redshift 
             (and clipped galaxy redshifts)
    '''

    galZs =np.array(galZs)

    if(binZ):
        # bin data into four large bins, and consider the most populated bin to be most likely
        # to contain members. Calculate cluster redshift and distribution center around only
        # the values in this bin.
        bins = np.linspace(min(galZs), max(galZs), 4)
        binPops = np.histogram(galZs, bins)[0]
        binPlaces = np.digitize(galZs, bins)
        binnedZs = galZs[binPlaces == np.argmax(binPops)+1]

        # Calculate cluster redshifts (z), or the biweight averages of the galaxy redshifts
        # (also called the biweight location estimator in Beers et al. (1990)
        center = stat.bAverage(binnedZs)
        center_err = stat.bootstrap_bAverage_err(binnedZs)
    else:
        center = stat.bAverage(galZs)
        center_err = stat.bootstrap_bAverage_err(galZs)

    initialZs = galZs

    # calculate galaxy velocities of all cluster members, using cluster redshift relation
    # remove all galaxies above 4,000 km/s from the center (or other specified vCut)
    # (galVs is returned in km/s by LOS_properVelocity)
    galVs = np.array((LOS_properVelocity(galZs, center))) 

    vMask = abs(galVs) <= vCut
    galZs = galZs[vMask]
    galVs = galVs[vMask]
    clippedMembers = 1 # to start while
    prevCount = len(initialZs)

    while (clippedMembers > 0):
        # Clip interloper galaxies
        memberVs = stat.sigmaClip(galVs)
        clipMask = memberVs.mask
        galZs = galZs[~clipMask]
        z = stat.bAverage(galZs)
        galVs = LOS_properVelocity(galZs, z)
        clippedMembers = prevCount - len(galZs)
        prevCount = len(galZs)
    
    z = stat.bAverage(galZs)
    z_err = stat.bootstrap_bAverage_err(galZs)
    if(len(galZs_err) != 0):
        [galVs, galVs_err] = LOS_properVelocity(initialZs, z, galZs_err, z_err)
    else:
        galVs = LOS_properVelocity(initialZs, z)
    mask = np.array([z in galZs for z in initialZs])

    if(len(galZs_err) == 0):
        return [galVs, z, z_err, mask]
    else:
        return [galVs, galVs_err, z, z_err, mask]


def vCut(galZs, centerZ = None, vCut = 4000, getZs = False):
    '''
    Preform just a velocity cut on a set of galaxies, do not clip. Useful if your dataset is very small.
    Also find galaxy velocities and cluster redshift, though with very few members (length of reshift array),
    the cluster redshift cannot be very reliable.
    :param galZs: array-like of galaxy redshifts
    :param vCut: velocity in km/s to cut at
    :param getZs: whether ot not to return clipped array of galaxy redshifts (mask should be enough)
    :return: boolean clip mask, galaxy velocities, and cluster redshift (and clipped galaxy redshifts)
    '''

    if(centerZ == None):
        # bin data into four large bins, and consider the most populated bin to be most likely
        # to contain members. Calculate cluster redshift and distribution center around only
        # the values in this bin.
        bins = np.linspace(min(galZs), max(galZs), 4)
        binPops = np.histogram(galZs, bins)[0]
        binPlaces = np.digitize(galZs, bins)
        binnedZs = galZs[binPlaces == np.argmax(binPops)+1]

        # Calculate cluster redshifts (z), or the biweight averages of the galaxy redshifts
        # (also called the biweight location estimator in Beers et al. (1990)
        centerZ = stat.bAverage(binnedZs)

    galVs = np.array((LOS_properVelocity(galZs, centerZ))) / 1000
    vMask = np.bitwise_and((galVs >= -vCut),(galVs <= vCut))
    galVs = galVs[vMask]
    galZs = galZs[vMask]

    try: biwtZ = stat.bAverage(galZs)
    except ZeroDivisionError: biwtZ = float('nan')

    if getZs:
        return [galVs, galZs, biwtZ, vMask]
    else:
        return [galVs, biwtZ, vMask]


def shifting_gapper(galVs, galCoords, centerZ, centerCoords, n=10, vCut = 1000):
    '''
    Preform the "shifting gapper" alogirthm of interloper removal, as described in Crawford et al. 2014.
    1. Iterate through each galaxy, finding n closest galaxies in clustocentric radial space
    2. Keep all of those n closest galaxies whose velocities are less than the target galaxy
    3. Find gaps between target galaxy and all these gathered galaxies
    4. If max gap is less than desired threshold, galaxy is a member

    :param galVs: array or array-like of galaxy velocities
    :param galCoords: array or array-like of galaxy coordinate tuples in form (RA,DEC)
    :param centerZ:  redshift of the cluster
    :param centerCoords: tuple of the cluster center coordinate in form (RA,DEC)
    :param n: number of closet galaxies to consider for each target galaxy
    :param vCut: velocity threshold on which to base membership criteria
    :return: memberVs, all velocities from input galVs which have been found to lie in the cluster
    '''

    memberMask = np.zeros(len(galVs))
    # convert coordinate data to radial separation
    distances = projectedDist(galCoords, centerCoords, centerZ)
    print(distances)

    # loop through all galaxies
    for i in distances:
        #find nearest n galaxies in radial space, mask out those with smaller velocities, and find gaps
        sep_order = np.argsort(abs(distances - i))
        nearestVs = galVs[sep_order][1:n+1]
        mask = abs(nearestVs) < abs(galVs[i])
        gaps = [ abs(galVs[i] - nearestVs[j]) for j in abs(nearestVs[mask]) ]

        if(max(gaps) < vCut):
            # max gap is below threshold; galaxy is considered a cluster member
            memberMask[i] = 1

    memberVs = galVs[memberMask]
    return memberVs


def NFW_escape_interlopers(radii, vel, r200, m200, z, c=6, cosmo=WMAP9):
    '''
    Preform interloper removal by cutting all galaxies that fall above the NFW-derived radial-dependent 
    escape velocity curve for a given cluster.
    
    :param radii: clustocentric radial distances for each galaxy in Mpc/h
    :param vel: LOS velocities for each galaxy in km/s
    :param r200: r200 radius for the host cluster in Mpc/h
    :param m200: m200 mass estimate for the host cluster in M_sun/h
    :param z: redshift of the host cluster
    :param c: concentration to assume (default c=6)
    :param cosmo: astropy cosmology instance (default is cosmo=Plack15)
    :return: the interloper mask, and the escape velocity lambda function, as a function of radius only
    '''

    # cast r200 to meters, and m200 to kg, and velocities to m/s
    r200 = r200 * 3.086e22
    radii = radii * 3.086e22
    m200 = m200 * const.M_sun.value
    vel = np.array(vel) * 1e3
    #v_disp = v_disp*1e3 / (cosmo.H(z)/100)

    s = lambda r,r200: r/r200
    gc = lambda c: 1/(np.log(1+c) - c/(1+c))
    k = lambda c,r,r200: gc(c) * np.log(1+c*s(r,r200))/s(r,r200)
    v_esc_NFW = lambda c,r,r200,m200: np.sqrt( (2*const.G.value*m200)/r200 * k(c, r, r200))
    v_esc = lambda r: v_esc_NFW(c, r, r200, m200)

    interloper_mask = np.array([v_esc(radii[i]) > vel[i] for i in range(len(radii))])
    return [interloper_mask, v_esc]
    
