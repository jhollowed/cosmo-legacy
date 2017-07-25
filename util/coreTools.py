"""
Joe Hollowed
Last edited 1/17/2017

Module containing tools for preforming analysis on core catalogs
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dispersionStats import bDispersion
import numpy as np
import pdb


def core_velDisp(cores, dim=3):
    """
    Calculate the velocity dispersion of a group of cores
    
    :param cores: numpy rec array with labeled fields, containing all cores 
                  in a particular halo (halo tag in file name) 
    :param dim: dimensions in which to calculate dispersion (defult = 3)
                (valid inputs are 3 or 1)
    :return: velocity dispersion in km s^-1
    """
    if dim == 3:
        v = np.array([np.linalg.norm([c['vx'], c['vy'], c['vz']], axis=0) for c in cores])
    elif dim == 1:
        v = cores['vx']
    else:
        raise ValueError('Argument dim can be 1 or 3')
    
    velDisp = bDispersion(v)
    return velDisp


def mask_tags(tags):
    """
    For all negative core tags, mask the first 16 bits to obtain original 
    halo tags of fragment halos
    
    :param tags: array of core halo tags to mask
    :return: recovered tag
    """
    mask = 281474976710655
    return abs(tags) & mask


def unwrap_position(pos, center, boxL=256):
    """
    Function to adjust positions of simulation particles/objects (cores)
    positions that have wrapped around simulation box due to periodic boundaries.
    
    :param pos: position(s) of n particle(s) in form 
                [[x0,y0,z0],...,[xn,yn,zn]] (in Mpc)
    :param center: relative center position in form [x, y, z] (in Mpc)
    :param boxL: length of simulation box in Mpc (default is 256)
    :return: unwrapped position (either <0 or >box length) in Mpc
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


def process_cores(cores, printarg=False, mass_cut=10**11.26, disrupt_rad=0.05):
    """
    This function processes the core (first arg) according to the last three of the
    following parameters:
    
    :param cores: a numpy rec array of cores
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
        print 'Halo has {} cores'.format(len(cores))
    
    cores = cores[mask]
    N = len(cores)
    if printarg:
        print 'Halo has {} cores after cuts'.format(N)
    
    return cores
