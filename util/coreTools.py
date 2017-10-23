"""
Joe Hollowed
Last edited 1/17/2017

Module containing tools for preforming analysis on core catalogs
"""
import pdb
import math
import numpy as np
import dispersionStats as stat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core import umath_tests as npm


# ==================================================================================================


def core_velDisp(cores, err = False, dim=3):
    """
    Calculate the velocity dispersion of a group of cores
    
    :param cores: numpy rec array with labeled fields, containing all cores 
                  in a particular halo (halo tag in file name) 
    :param err: whether or not to return the bootstrapped error of the biweight dispersion
                estimator
    :param dim: dimensions in which to calculate dispersion (defult = 3)
                (valid inputs are 3 or 1)
    :return: velocity dispersion in km s^-1 (and error if err==True)
    """
    if dim == 3:
        v = np.array([np.linalg.norm([c['vx'], c['vy'], c['vz']], axis=0) for c in cores])
    elif dim == 1:
        v = cores['vx']
    else:
        raise ValueError('Argument dim can be 1 or 3')
    
    [velDisp, velDisp_err] = stat.bootstrap_bDispersion(v, size = math.ceil(len(v)*0.75), repl=False)
    if(err):
        return [velDisp, velDisp_err]
    else:
        return velDisp


# ==================================================================================================


def mask_tags(tags):
    """
    For all negative core tags, mask the first 16 bits to obtain original 
    halo tags of fragment halos
    
    :param tags: array of core halo tags to mask
    :return: recovered tag
    """
    mask = 281474976710655
    return abs(tags) & mask


# ==================================================================================================


def get_velocity_components(x, y, z, vx, vy, vz):
    '''
    This function takes the positions and velocities of simulation objects within a halo,
    and returns their total, radial, and tangential velocity magnitudes, with respect to the
    halo center (defined as the median of the passed positions)

    :param x: an array of the comoving x-coordinates in Mpc/h
    :param y: an array of the comoving y-coordinates in Mpc/h
    :param z: an array of the comoving z-coordinates in Mpc/h
    :param vx: an arrray of the x-component of the velocities in comoving km/s
    :param vy: an arrray of the y-component of the velocities in comoving km/s
    :param vz: an arrray of the z-component of the velocities in comoving km/s
    :return: the total velocity magnitude, the radial component, and the tangential component
    '''
    n = len(x)
    r = np.array([x, y, z]).T
    v = np.array([vx, vy, vz]).T
    if(n > 100):
        halo_center = np.median(r.T, axis=1)
        halo_vel = np.median(v.T, axis=1)
    else:
        halo_center = [stat.bAverage(p) for p in [x, y, z]]
        halo_vel = [stat.bAverage(vp) for vp in [vx, vy, vz]]
    
    # get positions and velocities in halocentric frame 
    r_rel = r - halo_center
    v_rel = v - halo_vel
    v_rel_mag = np.linalg.norm(v_rel, axis=1)
    
    # get radial velocity component
    r_rel_mag = np.linalg.norm(r_rel, axis=1)
    r_rel_hat = np.divide(r_rel, np.array([r_rel_mag]).T)
    v_rad = npm.inner1d(v_rel, r_rel_hat)
     
    # get tangential velocity component
    tan_dir = np.cross( np.cross(v_rel, r_rel, axisa=1), r_rel, axisa=1)
    tan_dir_mag = np.linalg.norm(tan_dir, axis=1)
    tan_dir_hat = np.divide(tan_dir, np.array([tan_dir_mag]).T)
    v_tan = npm.inner1d(v_rel, tan_dir_hat)
    
    # consistency check
    v_rel_mag_check = np.sqrt(v_rad**2 + v_tan**2)
    diff = [stat.percentDiff(v_rel_mag_check[k], v_rel_mag[k]) for k in range(len(v))]
    if( max(diff) > 0.001):
        raise ValueError('component quad is not recovering velocity magnitude (bug)')
    
    return v_rel_mag, v_rad, v_tan


# ==================================================================================================


def get_radial_distance(x, y, z):
    '''
    This function takes the comiving positions of simulation objects within a halo, and gives
    the radial distance of each object to the halo center in Mpc/h

    :param x: an array of the comoving x-coordinates in Mpc/h
    :param y: an array of the comoving y-coordinates in Mpc/h
    :param z: an array of the comoving z-coordinates in Mpc/h
    '''

    n = len(x)
    r = np.array([x, y, z]).T
    if(n > 100):
        halo_center = np.median(r.T, axis=1)
    else:
        halo_center = [stat.bAverage(p) for p in [x, y, z]]
    
    r_rad = r - halo_center
    r_rad_mag = np.linalg.norm(r_rad, axis=1)
    return r_rad_mag


# ==================================================================================================


def unwrap_position(pos, center, boxL=256):
    """
    Function to adjust positions of simulation particles/objects (cores)
    positions that have wrapped around simulation box due to periodic boundaries.
    
    :param pos: position(s) of n particle(s) in form 
                [[x0,y0,z0],...,[xn,yn,zn]] (in Mpc/h)
    :param center: relative center position in form [x, y, z] (in Mpc/h)
    :param boxL: length of simulation box in Mpc (default is 256)
    :return: unwrapped position (either <0 or >box length) in Mpc/h
    """
    for n in range(len(pos)):
        p = pos[n]
        for i in range(3):
            dist = p[i] - center[i]
            if dist > boxL / 2.0:
                pos[n][i] = p[i] - boxL
            elif dist < -boxL / 2.0:
                pos[n][i] = p[i] + boxL

    return pos


# ==================================================================================================


def process_cores(masses, radii, printarg=False, mass_cut=10**11.27, disrupt_rad=0.05):
    """
    This function processes the core (first arg) according to the last three of the
    following parameters:
    
    :param masses: numpy array of core infall masses
    :param radii: numpy array of core radii
    :param printarg: whether or not to print function progress
    :param mass_cut: cores below this infall_mass value will be discarded
    :param disrupt_rad: cores with radii above this value will be discarded
    :return: the processed list of cores
    
    The default values for these parameters were found by fitting to SDSS galaxy profiles
    by Dan. They are
    mass_cut = 10**11.26 M_sun/h
    disrupt_rad = 0.05 Mpc/h
    """
    mass_mask = cores['infall_mass'] > mass_cut
    rad_mask = cores['radius'] < disrupt_rad
    mask = np.logical_and(mass_mask, rad_mask)
    if printarg:
        print('Halo has {} cores'.format(len(cores)))
    
    cores = cores[mask]
    N = len(cores)
    if printarg:
        print('Halo has {} cores after cuts'.format(N))
    
    return cores
