# uncompyle6 version 2.9.11
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.5.2 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:53:06) 
# [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
# Embedded file name: /home/jphollowed/code/sim_scripts/read/coreTools.py
# Compiled at: 2017-04-04 13:53:51
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


def process_cores(cores, merge=False, printarg=False, mass_cut=398107170553.49695, disrupt_rad=0.3, merger_dist=0.007):
    """
    This function processes the core (first arg) according to the last three of the
    following parameters:
    
    :param cores: a numpy rec array of cores
    :param merge: whether or not to preform merging of the cores
    :param printarg: whether or not to print function progress
    :param mass_cut: cores below this infall_mass value will be discarded
    :param disrupt_rad: cores with radii above this value will be discarded
    :param merger_dist: any cores within this distance of eachother will be merged
    :return: the processed list of cores
    
    The default values for these parameters were found by fitting to SDSS galaxy profiles
    by Dan. They are
    mass_cut = 10**11.6 M_sun/h
    disrupt_rad = 0.3 Mpc/h
    merger_dist = 0.007 Mpc/h
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
    
    if not merge:
        return cores
    
    else:
        core_positions = np.swapaxes([ cores[r] for r in ['x', 'y', 'z'] ], 0, 1)
        core_dist = np.linalg.norm(core_positions, axis=1)
        indices = np.linspace(0, N - 1, N).astype(int)
        dist_matrix = np.array([ abs(core_dist[n] - core_dist) for n in range(N) ])
        dist_matrix = np.triu(dist_matrix)
        merger_matrix = np.logical_and(dist_matrix != 0, dist_matrix < merger_dist) + np.eye(N).astype(bool)
        merged_core_members = []
        checked = []
        for n in range(len(merger_matrix)):
            if n in checked:
                continue
            core_pairs = merger_matrix[n]
            for j in range(len(core_pairs)):
                if j in checked:
                    continue
                if core_pairs[j]:
                    checked.append(j)
                    core_pairs = np.logical_or(core_pairs, merger_matrix[j])

            merged_core_members.append(indices[core_pairs])

        merged_cores = np.array([ cores[merged_core_members[n]] for n in range(len(merged_core_members))
                                ])
        final_cores = np.empty(len(merged_cores), dtype=cores.dtype)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(merged_cores)):
            final_core = merged_cores[i][0]
            for column in merged_cores[i].dtype.names[2:]:
                final_core[column] = np.mean(merged_cores[i][column])

            final_core['infall_mass'] = sum(merged_cores[i]['infall_mass'])
            final_cores[i] = final_core
            ax.plot(merged_cores[i]['x'], merged_cores[i]['y'], merged_cores[i]['z'], 'x', ms=8, color=np.random.rand(3))
            ax.plot([final_core['x']], [final_core['y']], final_core['z'], '.r', ms=10)

        plt.show()
        if printarg:
            print 'Halo has {} cores after merging'.format(len(final_cores))
        return final_cores
