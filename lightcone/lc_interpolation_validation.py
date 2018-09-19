
import pdb
import glob
import h5py as h5
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import WMAP7 as wmap7
import matplotlib as mpl
from cycler import cycler
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm

'''
This file contains functions for inspecting and visualizing lightcone output
'''

#######################################
#
#             Utility functions
#
#######################################

def config(cmap, numColors=3):
    '''
    This function prepares the matplotlib plotting environment in
    my perferred way. It sets the plotting color cycle to be a 
    discrete sampling of a desired colormap, and does some other
    formatting.

    Params:
    :param cmap: The colormap to use to build a new color cycle
    :param numColors: The length of the color cycle. The colormap 
                      cmap will be sampled at an interval of
                      1/numColors to populate the color cycle. 
    :return: None
    '''
    
    rcParams.update({'figure.autolayout': True})
    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    rcParams.update(params)
    colors = cmap(np.linspace(0.2, 0.8, numColors))
    c = cycler('color', colors)
    plt.rcParams["axes.prop_cycle"] = c
    

def plotBox(x, y, z, xw, yw, zw, ax, color, alpha=1):
    '''
    Plot a three dimensional rectangular prism, with the position 
    of its corner nearest the origin, and prism side widths, specified. 

    Params:
    :param x: The x-position of the prism corner nearest the origin
    :param y: The y-position of the prism corner nearest the origin
    :param z: The z-position of the prism corner nearest the origin
    :param xw: The width of the prism in the x-dimension
    :param yw: The width of the prism in the y-dimension
    :param zw: The width of the prism in the z-dimension
    :return: None
    '''

    xx, yy = np.meshgrid([x, x+xw], [y, y+yw])
    ax.plot_surface(xx, yy, np.ones(np.shape(xx))*z, color=color, alpha=alpha, shade=False)
    ax.plot_surface(xx, yy, np.ones(np.shape(xx))*(z+zw), color=color, alpha=alpha, shade=False)

    yy, zz = np.meshgrid([y, y+yw], [z, z+zw])
    ax.plot_surface(np.ones(np.shape(yy))*x, yy, zz, color=color, alpha=alpha, shade=False)
    ax.plot_surface(np.ones(np.shape(yy))*(x+xw), yy, zz, color=color, alpha=alpha, shade=False)

    zz, xx = np.meshgrid([z, z+zw], [x, x+xw])
    ax.plot_surface(xx, np.ones(np.shape(xx))*y, zz, color=color, alpha=alpha, shade=False)
    ax.plot_surface(xx, np.ones(np.shape(xx))*(y+yw), zz, color=color, alpha=alpha, shade=False)


def pair(k1, k2, safe=True):
    """
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    """
    z = int(0.5 * (k1 + k2) * (k1 + k2 + 1) + k2)
    if safe and (k1, k2) != depair(z):
        raise ValueError("{} and {} cannot be paired".format(k1, k2))
    return z


def depair(z):
    """
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    w = np.floor((np.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    # assert z != pair(x, y, safe=False):
    return x, y


def search_sorted(array, values, sorter=None):
    '''
    Find the positon of values within an array via a binary search. This 
    simply implements numpy's searchsorted function, but forces values that 
    are *not* present in the searched array to return an index of -1. 
    This function originally from dtk.

    Params:
    :param array: Input array. If sorter is None, then it will be sorted in 
                  ascending order, otherwise sorter must be an array of indices 
                  that sort it.
    :param values: Values to find in array
    :param sorter: Optional array of integer indices that sort array into ascending
                   order. This is typically the result of an argsort
    '''
    
    if sorter is None:
        sorter = np.argsort(array)
    
    start = np.atleast_1d(np.searchsorted(array,values,side='left',sorter=sorter))
    slct_start = start == array.size
    start[slct_start] = 0;
    
    start = sorter[start]
    slct_match = np.atleast_1d(array[start]==values)
    start[slct_match==0]=-1
    return start


def unit_vector(vector):
    '''
    Normalizes an input vector.
    '''
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    '''
    Returns the angle in radians between vectors 'v1' and 'v2'
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#############################################################################################


#######################################
#
#           Validation Tests
#
#######################################

def lightconeHistograms(lcDir1, lcDir2, step, rL, relativeError = True, mode='particles', 
                        plotMode='show', outDir='.'):
    '''
    This function plots the difference in particle/object position, velocity, and
    redshift, between two different lightcone runs, as histograms

    Params:
    :param lcDir1: path to a lightcone output directory It is assumed that this 
                   directory follows the naming and structure convention given in 
                   fig 6 of the Creating Lightcones in HACC notes (with one 
                   subdirectory per snapshot).
    :param lcDir2: path toanother lightcone output directory, to compare with the 
                   output at lcDir1. This lightone put should have been run with 
                   identical parameters, on the same cimulation volume, as the run 
                   that generated the data at lcDir1. The same directory structure 
                   as explained in the docstring for arg lcDir1 is assumed here as well.
    :param step: teh lightcone output step at which to perform the comparison
    :param rL: the box width of the simulation from which the lightcones give in lcDir1
               and lcDir2 were generated
    :param mode: whether to perform the snapshot object match-up on particles or
                 halos. If mode=="particles", then find the object in each snapshot
                 by matching on it's 'id'. If mode=="halos", then find the object
                 in each snapshot by matching it's 'tree_node_index' and 
                 'desc_node_index'.
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless otherwise
                     specified in the 'outDir' arg
    :param outDir: where to save figures, if plotMode == 'save'
    :return: None
    '''
    if plotMode not in ['show', 'save']:
        raise Exception('Unknown plotMode {}. Options are \'show\' or \'save\'.'.format(plotMode))
    if mode not in ['particles', 'halos']:
        raise Exception('Unknown mode {}. Options are \'particles\' or \'halos\'.'.format(mode))

    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio
    
    subdirs = glob.glob('{}/*'.format(lcDir1))
    
    # get lc subdirectory prefix (could be 'lc' or 'lcGals', etc.). 
    # prefix of subdirs in lcDir2 and lcDir1 assumed to be the same.
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue

    # get file names in 442 subdir for interpolated and extrapolated lc data
    # (sort them to grab only the unhashed file header)
    file1 = sorted(glob.glob('{}/{}{}/*'.format(lcDir1, prefix, step)))[0]
    file2 = sorted(glob.glob('{}/{}{}/*'.format(lcDir2, prefix, step)))[0]
   
    # set id types to read based on the mode
    if(mode == 'particles'):
        idName = 'id'
    elif(mode == 'halos'):
        idName = 'tree_node_index'
        
    # read data
    print("Reading files from {}".format(lcDir1.split('/')[-1]))
    iid = gio.gio_read(file1, 'id')
    ix = gio.gio_read(file1, 'x')
    iy = gio.gio_read(file1, 'y')
    iz = gio.gio_read(file1, 'z')
    ivx = gio.gio_read(file1, 'vx')
    ivy = gio.gio_read(file1, 'vy')
    ivz = gio.gio_read(file1, 'vz')
    ia = gio.gio_read(file1, 'a')
    irot = gio.gio_read(file1, 'rotation')

    print("Reading files from {}".format(lcDir2.split('/')[-1]))
    eid = gio.gio_read(file2, 'id')
    ex = gio.gio_read(file2, 'x')
    ey = gio.gio_read(file2, 'y')
    ez = gio.gio_read(file2, 'z')
    evx = gio.gio_read(file2, 'vx')
    evy = gio.gio_read(file2, 'vy')
    evz = gio.gio_read(file2, 'vz')
    ea = gio.gio_read(file2, 'a')
    erot = gio.gio_read(file2, 'rotation')

    # get rid of everything not in the initial volume (don't
    # consider objects found in replicated boxes, so matchup
    # is simplified)

    # decrease simulation box side length value by 1% to avoid
    # grabbing objects who originate from some other box replication,
    # but moved into rL by the lightcone position approximation
    rL = rL * 0.99

    initVolMask1 = np.logical_and.reduce((abs(ix) < rL, 
                                               abs(iy) < rL, 
                                               abs(iz) < rL))
    iid = iid[initVolMask1]
    ix = ix[initVolMask1]
    iy = iy[initVolMask1]
    iz = iz[initVolMask1]
    ia = ia[initVolMask1]
    irot = irot[initVolMask1]

    initVolMask2 = np.logical_and.reduce((abs(ex) < rL, 
                                               abs(ey) < rL, 
                                               abs(ez) < rL))
    eid = eid[initVolMask2]
    ex = ex[initVolMask2]
    ey = ey[initVolMask2]
    ez = ez[initVolMask2]
    ea = ea[initVolMask2]
    erot = erot[initVolMask2]

    # make sure that worked...
    if(len(np.unique(irot)) > 1 or len(np.unique(erot)) > 1):
        raise Exception('particles found in replicated boxes >:(')
    
    # find unique objects to begin matching 
    print('finding unique')
    iunique = np.unique(iid, return_counts=True)
    eunique = np.unique(eid, return_counts=True)
    if(max(iunique[1]) > 1 or max(eunique[1]) > 1): 
        # There were duplicates found in this volume. pdb trace?
        pass
    
    # get rid of duplicates in interpolation lc data
    iuniqueMask = np.ones(len(iid), dtype=bool)
    iuniqueMask[np.where(np.in1d(iid, iunique[0][iunique[1] > 1]))[0]] = 0

    print('get intersecting data (union of interp and extrap lc objects)')
    intersection_itoe = np.in1d(iid[iuniqueMask], eid)
    intersection_etoi = np.in1d(eid, iid[iuniqueMask])

    print('sorting extrap data array by id to match order of interp data array')
    eSort = np.argsort(eid[intersection_etoi])

    # do binary search using to find each object from the interpolation
    # lc data in the extrapolation lc data
    print('matching arrays')
    matchMap = search_sorted(eid[intersection_etoi], 
                                  iid[iuniqueMask][intersection_itoe], sorter=eSort)

    iMask = np.linspace(0, len(iid)-1, len(iid), dtype=int)[iuniqueMask][intersection_itoe]
    eMask = np.linspace(0, len(eid)-1, len(eid), dtype=int)[intersection_etoi][matchMap]

    print('diffing positions')
    xdiff = np.squeeze(ix[iMask] - ex[eMask])
    ydiff = np.squeeze(iy[iMask] - ey[eMask])
    zdiff = np.squeeze(iz[iMask] - ez[eMask])
    posDiff = np.linalg.norm(np.array([xdiff, ydiff, zdiff]).T, axis=1) 
    if(relativeError and 0):
        ipos = np.linalg.norm(np.array([np.squeeze(ix[iMask]), 
                                        np.squeeze(iy[iMask]), 
                                        np.squeeze(iz[iMask])]).T, axis=1)
        zeroMask = ipos != 0
        posDiff = posDiff[zeroMask] / ipos[zeroMask]
        mean_posDiff = np.mean(posDiff)
        std_posDiff = np.mean(posDiff)
        print('Relative positional difference is {} +- {}'.format(mean_posDiff, std_posDiff))
    
    print('diffing velocities')
    vxdiff = np.squeeze(ivx[iMask] - evx[eMask])
    vydiff = np.squeeze(ivy[iMask] - evy[eMask])
    vzdiff = np.squeeze(ivz[iMask] - evz[eMask])
    mag_vDiff = np.linalg.norm(np.array([vxdiff, vydiff, vzdiff]).T, axis=1)
    if(relativeError):
        mag_iv = np.linalg.norm(np.array([np.squeeze(ivx[iMask]), 
                                          np.squeeze(ivy[iMask]), 
                                          np.squeeze(ivz[iMask])]).T, axis=1)
        zeroMask = mag_iv != 0
        mag_vDiff = mag_vDiff[zeroMask] / mag_iv[zeroMask]
        mean_vDiff = np.mean(mag_vDiff)
        std_vDiff = np.mean(mag_vDiff)
        print('Relative velocity difference is {} +- {}'.format(mean_vDiff, std_vDiff))

    print('diffing redshift')
    redshiftDiff = np.abs(((1/ia)-1)[iMask] - ((1/ea)-1)[eMask])
    if(relativeError and 0):
        iredshift = ((1/ia)-1)[iMask]
        redshiftDiff = redshiftDiff / iredshift
        mean_redshiftDiff = np.mean(posDiff)
        std_redshiftDiff = np.mean(posDiff)
        print('Relative redshift difference is {} +- {}'.format(mean_redshiftDiff, std_redshiftDiff))
 
    # plot position, velocity, and redshift differences between interpolated
    # and extrapolated output as historgrams

    config(cmap=plt.cm.plasma)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bins = 300
    
    f = plt.figure(plt.gcf().number+1) 
    ax =  f.add_subplot(311)
    ax.hist(posDiff, bins, color=colors[0], log=True)
    ax.set_ylim(ymin=1)
    ax.set_xlabel(r'$\left|\vec{r}_\mathrm{extrap} - \vec{r}_\mathrm{interp}\right|\>\>\mathrm{(Mpc/h)}$', fontsize=18)
    
    ax2 =  f.add_subplot(312)
    ax2.hist(mag_vDiff, bins, color=colors[1], log=True)
    ax2.set_ylim(ymin=1)
    ax2.set_xlabel(r'$\left|\vec{v}_\mathrm{extrap} - \vec{v}_\mathrm{interp}\right| \>\>\mathrm{(km/s)}$', fontsize=18)
    
    ax3 =  f.add_subplot(313)
    ax3.hist(redshiftDiff, bins, color=colors[2], log=True)
    ax3.set_ylim(ymin=1)
    ax3.set_xlabel(r'$\left|z_\mathrm{extrap} - z_\mathrm{interp}\right|$', fontsize=18)
    
    if(plotMode == 'show'):
        plt.show()
    else:
        plt.savefig('{}/lc_diffHists_{}'.format(outDir, step))


#############################################################################################
#############################################################################################


def saveLightconePathData(epath, ipath, spath, outpath, rL, lcStep=442, diffRange='max', 
                          mode='particles', fullParticles = False, solverMode = 'forward', 
                          snapshotSubdirs = False, fragmentsOnly=False):
    '''
    This function loads lightcone output data, and inspects the 
    difference in position resulting from the extrapolation and interpolation
    routines. This difference, per lc object, is called diffVals. It then saves 
    the 3 dimensional paths of 10 of these particles-- either the 10 with the 
    largest diffVals, the median, or minimum-- as .npy files. The results can 
    be viewed as 3d plots by calling plotLightconePaths() below.
    This 3d path includes the true path of the particle from surrounding snapshots, 
    where the particle from the lightcone is matched to the snapshots by id 
    (using descendent/parent ids in the case that the input data is a halo
    lightcone, and merger trees need to be used), and the approximated path
    given by the lightcone solver.
    
    For consistency, the time window used by default to inspect the objects paths is
    always steps 421 - 464, with step 442 being the one to read lightcone data
    from. If the lightcone step to inspect is specified as otherwise, then the time 
    window visualized will pad that lightcone step with up to two snapshot tiemstep 
    outputs on either side. It is assumed that the lightcone output directories follow 
    the naming and structure convention given in fig 6 of the Creating Lightcones
    in HACC notes. The strcture of the snapshot data directory can be given by
    the user in the snapshotSubdirectories parameter.
    
    This code is meant to be run on Datastar, or Cooley (somewhere with gio compiled).
    
    Params:
    :param epath: path to a lightcone output directory generated by the 
                  extrapolation driver
    :param ipath: path to a lightcone output directory generated by the
                  interpolation driver. Should have been run with identical 
                  parameters, on the same cimulation volume, as the run that 
                  generated the data at epath
    :param spath: path to top-level directory of snapshot data from the simulation 
                  that was used to generate the data at epath and ipath. If mode=='halos', 
                  this should be the path to the top-level directory of merger tree data
    :param outpath: where to write out the path data (npy files)
    :param rL: the box width of the simulation from which the lightcones at epath
               and ipath were generated
    :param lcStep: the lightcone step output to use as the approximate trajectory data
    :param diffRange: whether to use the 'max', 'med'(median) or 'min' diffVals
               and ipath were generated, in comoving Mpc/h
    :param mode: whether to perform the snapshot object match-up on particles or
                 halos. If mode=="particles", then find the object in each snapshot
                 by matching on it's 'id'. If mode=="halos", then find the object
                 in each snapshot by matching it's 'tree_node_index' and 
                 'desc_node_index'.
    :param fullParticles: whether or not the input lightcones were run on full particle snapshots
                          (dont attempt to run this test if this arg is True, but the snapshot files
                           are enormous)
    :param solverMode: Wether the lightcone solver used to generate the output under 
                       validation was run in the standard forward mode (solverMode = 
                       'forward'), or the reversed mode, with extrapolation/interpolation
                       occuring backward in time (solverMode = 'backward')
    :param snapshotSubdirs: If true, assume that the snapshot data is grouped into
                            subdirectories of the format "STEPXXX". Otherwise, 
                            assume one flat directory with the the step number
                            "XXX" in the filenames somewhere, as "*.XXX.*" where
                            * is a wildcard
    :param fragmentsOnly: Whether or not to inspect the trajectories of fragment halos
                          only, in the case that mode == 'halos'
    :return: None
    '''

    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio
    
    if mode not in ['particles', 'halos']:
        raise Exception('Unknown mode {}. Options are \'particles\' or \'halos\'.'.format(mode))
    if solverMode not in ['forward', 'backward']:
        raise Exception('Unknown solver mode {}. Options are \'forward\' or \'backward\'.'.format(solverMode))
    if diffRange not in ['min', 'med', 'max']:
        raise Exception('Unknown diffRange {}. Options are \'min\', \'med\', or \'max\'.'
                        .format(diffRange))
    
    subdirs = glob.glob('{}/*'.format(ipath))

    # get time window to save (attempt to pad input lightcone step with two snapshot timesteps
    # on either side)
    if(snapshotSubdirs):
        snapshot_step_dirs = np.array(glob.glob('{}/STEP*'.format(spath)))
        steps_avail = np.array(sorted([int(s.split('STEP')[-1]) for s in snapshot_step_dirs]))
    else:
        snapshot_files = np.array(glob.glob('{}/*.mpicosmo.*'.format(spath)))
        snapshot_files = snapshot_files[[not '#' in ss for ss in snapshot_files]]
        if(not fullParticles):
            snapshot_files = snapshot_files[[not 'full' in ss for ss in snapshot_files]]
        else:
            snapshot_files = snapshot_files[['full' in ss for ss in snapshot_files]]
        steps_avail = np.array(sorted([int(s.split('.')[-1]) for s in snapshot_files]))
    
    # remove corrupt outerRim steps
    bad_idx = list(steps_avail).index(304)
    steps_avail = np.delete(steps_avail, bad_idx)

    lcStep_idx = list(steps_avail).index(lcStep)
    if(lcStep_idx < 2 or lcStep_idx > len(steps_avail)-2): 
        raise Exception('specified lightcone step must have two snapshot timesteps available before and after it')
    traj_steps = steps_avail[lcStep_idx + np.array([-2, -1, 0, 1, 2])]

    print('time window will be {}'.format(traj_steps))
    
    # get lc subdirectory prefix (could be 'lc' or 'lcGals', etc.). 
    # prefix of subdirs in epath and ipath assumed to be the same.
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue

    # get file names in step subdir for interpolated and extrapolated lc data
    # (sort them to grab only the unhashed file header)
    ifile = sorted(glob.glob('{}/{}{}/*'.format(ipath, prefix, lcStep)))[0]
    efile = sorted(glob.glob('{}/{}{}/*'.format(epath, prefix, lcStep)))[0]
   
    # set id types to read based on the mode
    if(mode == 'particles'):
        idName = 'id'
    elif(mode == 'halos'):
        idName = 'tree_node_index'
  
    # read data
    print("Reading interpolation files")
    iid = gio.gio_read(ifile, 'id')
    ix = gio.gio_read(ifile, 'x')
    iy = gio.gio_read(ifile, 'y')
    iz = gio.gio_read(ifile, 'z')
    ivx = gio.gio_read(ifile, 'vx')
    ivy = gio.gio_read(ifile, 'vy')
    ivz = gio.gio_read(ifile, 'vz')
    ia = gio.gio_read(ifile, 'a')
    irot = gio.gio_read(ifile, 'rotation')

    print("Reading extrapolation files")
    eid = gio.gio_read(efile, 'id')
    ex = gio.gio_read(efile, 'x')
    ey = gio.gio_read(efile, 'y')
    ez = gio.gio_read(efile, 'z')
    evx = gio.gio_read(efile, 'vx')
    evy = gio.gio_read(efile, 'vy')
    evz = gio.gio_read(efile, 'vz')
    ea = gio.gio_read(efile, 'a')
    erot = gio.gio_read(efile, 'rotation')

    # get rid of everything not in the initial volume (don't
    # consider objects found in replicated boxes, since we have
    # no corresponding snapshot data there)
    
    # decrease simulation box side length value by 1% to avoid
    # grabbing objects who originate from some other box replication,
    # but moved into rL by the lightcone position approximation
    rL = rL * 0.99

    initVolMask_interp = np.logical_and.reduce((abs(ix) < rL, 
                                               abs(iy) < rL, 
                                               abs(iz) < rL))
    iid = iid[initVolMask_interp]
    if(fragmentsOnly):
        fragmentMask = iid < 0
    else:
        fragmentMask = np.ones(len(iid), dtype=bool)
    iid = iid[fragmentMask]
    
    ix = ix[initVolMask_interp][fragmentMask]
    iy = iy[initVolMask_interp][fragmentMask]
    iz = iz[initVolMask_interp][fragmentMask]
    ia = ia[initVolMask_interp][fragmentMask]
    irot = irot[initVolMask_interp][fragmentMask]


    initVolMask_extrap = np.logical_and.reduce((abs(ex) < rL, 
                                               abs(ey) < rL, 
                                               abs(ez) < rL))
    eid = eid[initVolMask_extrap]
    if(fragmentsOnly):
        fragmentMask = eid < 0
    else:
        fragmentMask = np.ones(len(eid), dtype=bool)
    eid = eid[fragmentMask]

    ex = ex[initVolMask_extrap][fragmentMask]
    ey = ey[initVolMask_extrap][fragmentMask]
    ez = ez[initVolMask_extrap][fragmentMask]
    ea = ea[initVolMask_extrap][fragmentMask]
    erot = erot[initVolMask_extrap][fragmentMask]

    # make sure that worked...
    if(len(np.unique(irot)) > 1 or len(np.unique(erot)) > 1):
        raise Exception('particles found in replicated boxes >:(')


    # find unique objects to begin matching 
    print('finding unique')
    iunique = np.unique(iid, return_counts=True)
    eunique = np.unique(eid, return_counts=True)
    if(max(iunique[1]) > 1 or max(eunique[1]) > 1): 
        # There were duplicates found in this volume. pdb trace?
        pass
    
    # get rid of duplicates in interpolation lc data
    iuniqueMask = np.ones(len(iid), dtype=bool)
    iuniqueMask[np.where(np.in1d(iid, iunique[0][iunique[1] > 1]))[0]] = 0

    print('get intersecting data (union of interp and extrap lc objects)')
    intersection_itoe = np.in1d(iid[iuniqueMask], eid)
    intersection_etoi = np.in1d(eid, iid[iuniqueMask])

    print('sorting extrap data array by id to match order of interp data array')
    eSort = np.argsort(eid[intersection_etoi])

    # do binary search using to find each object from the interpolation
    # lc data in the extrapolation lc data
    print('matching arrays')
    matchMap = search_sorted(eid[intersection_etoi], 
                                  iid[iuniqueMask][intersection_itoe], sorter=eSort)

    iMask = np.linspace(0, len(iid)-1, len(iid), dtype=int)[iuniqueMask][intersection_itoe]
    eMask = np.linspace(0, len(eid)-1, len(eid), dtype=int)[intersection_etoi][matchMap]

    print('diffing positions')
    xdiff = ix[iMask] - ex[eMask]
    ydiff = iy[iMask] - ey[eMask]
    zdiff = iz[iMask] - ez[eMask]
    posDiff = np.linalg.norm(np.array([xdiff, ydiff, zdiff]).T, axis=1) 
    
    print('diffing velocities')
    vxdiff = ivx[iMask] - evx[eMask]
    vydiff = ivy[iMask] - evy[eMask]
    vzdiff = ivz[iMask] - evz[eMask]
    mag_vDiff = np.linalg.norm(np.array([vxdiff, vydiff, vzdiff]).T, axis=1)[0]
    
    print('diffing redshift')
    redshiftDiff = np.abs(((1/ia)-1)[iMask] - ((1/ea)-1)[eMask])
 
    # find objects with position differences that are the max, median, or min of 
    # all object differences, depending on the argument given as diffRange. In the
    # case that diffRange='max', for instance, an array diffVals is created that 
    # contains the ten lightcone objects that have the largest difference between 
    # the extrapolation and interpolation solutions. Likewise for diffRange=med and
    # diffRange=min

    print('matching to specified range ({})'.format(diffRange))
    if(diffRange == 'max'):
        diffVals = np.argsort(posDiff)[::-1][0:10]
        savePath = '{}/max_diff'.format(outpath)
    if(diffRange == 'med'):
        diffVals = np.argsort(posDiff)[::-1][len(xdiff)/2:len(xdiff)/2 + 20][0:10]
        savePath = '{}/med_diff'.format(outpath)
    if(diffRange == 'min'):
        diffVals = np.argsort(posDiff)[0:10]
        savePath = '{}/min_diff'.format(outpath)

    # read snapshots...     
    # apply lightcone box rotations to snapshot data
    sx, sy, sz = 'x', 'y', 'z'
    if(irot[0] == 1): 
        sx, sy = 'y', 'x' 
    elif(irot[0] == 2): 
        sy, sz = 'z', 'y'
    elif(irot[0] == 3): 
        sx, sy = 'y', 'x'
        sy, sz = 'z', 'y'
    elif(irot[0] == 4):
        sz, sx = 'x', 'z'
    elif(irot[0] == 5):
        sx, sy = 'y', 'x'
        sz, sx = 'x', 'z'
    if(mode=='halos'):
        sx = 'fof_halo_center_{}'.format(sx)
        sy = 'fof_halo_center_{}'.format(sy)
        sz = 'fof_halo_center_{}'.format(sz)
    
    print("Reading snapshot files")
    # sort snapshot files and grad the first in the list so that we are
    # reading the un-hashed header file

    if(snapshotSubdirs): 
        sfiles0 = np.array(glob.glob('{}/STEP{}/*mpicosmo*'.format(spath, traj_steps[0])))
        # dont want haloparticles files or anything else like that
        sfiles0 = sfiles0[[not 'halo' in ss for ss in sfiles0]]
    else:
        sfiles0 = np.array(glob.glob('{}/*.{}*'.format(spath, traj_steps[0])))
        sfiles0 = sfiles0[[not 'halo' in ss for ss in sfiles0]]
    if(not fullParticles): 
        sfiles0 = sfiles0[[not 'full' in ss for ss in sfiles0]]
    else:
        sfiles0 = sfiles0[['full' in ss for ss in sfiles0]]
    sfile0 = sorted(sfiles0)[0]
    print('reading snapshot at {}'.format(sfile0))
    sid0 = np.squeeze(gio.gio_read(sfile0, idName))
    sx0 = np.squeeze(gio.gio_read(sfile0, sx))
    sy0 = np.squeeze(gio.gio_read(sfile0, sy))
    sz0 = np.squeeze(gio.gio_read(sfile0, sz))
    if(mode == 'halos'):
        # if halo mode is specified, read the descedent node index, 
        # fof tag
        sdid0 = np.squeeze(gio.gio_read(sfile0, 'desc_node_index'))
        stag0 = np.squeeze(gio.gio_read(sfile0, 'fof_halo_tag'))

    if(snapshotSubdirs): 
        sfiles1 = np.array(glob.glob('{}/STEP{}/*mpicosmo*'.format(spath, traj_steps[1])))
        sfiles1 = sfiles1[[not 'halo' in ss for ss in sfiles1]]
    else:
        sfiles1 = np.array(glob.glob('{}/*mpicosmo*.{}*'.format(spath, traj_steps[1])))
        sfiles1 = sfiles1[[not 'halo' in ss for ss in sfiles1]]
    if(not fullParticles): 
        sfiles1 = sfiles1[[not 'full' in ss for ss in sfiles1]]
    else:
        sfiles1 = sfiles1[['full' in ss for ss in sfiles1]]
    sfile1 = sorted(sfiles1)[0]
    print('reading snapshot at {}'.format(sfile1))
    sid1 = np.squeeze(gio.gio_read(sfile1, idName))
    sx1 = np.squeeze(gio.gio_read(sfile1, sx))
    sy1 = np.squeeze(gio.gio_read(sfile1, sy))
    sz1 = np.squeeze(gio.gio_read(sfile1, sz))
    if(mode == 'halos'):
        sdid1 = np.squeeze(gio.gio_read(sfile1, 'desc_node_index'))  
        stag1 = np.squeeze(gio.gio_read(sfile1, 'fof_halo_tag'))
        smass1 = np.squeeze(gio.gio_read(sfile1, 'fof_halo_mass'))
    
    if(snapshotSubdirs): 
        sfiles2 = np.array(glob.glob('{}/STEP{}/*mpicosmo*'.format(spath, traj_steps[2])))
        sfiles2 = sfiles2[[not 'halo' in ss for ss in sfiles2]]
    else:
        sfiles2 = np.array(glob.glob('{}/*mpicosmo*.{}*'.format(spath, traj_steps[2])))
        sfiles2 = sfiles2[[not 'halo' in ss for ss in sfiles2]]
    if(not fullParticles): 
        sfiles2 = sfiles2[[not 'full' in ss for ss in sfiles2]]
    else:
        sfiles2 = sfiles2[['full' in ss for ss in sfiles2]]
    sfile2 = sorted(sfiles2)[0]
    print('reading snapshot at {}'.format(sfile2))
    sid2 = np.squeeze(gio.gio_read(sfile2, idName))
    sx2 = np.squeeze(gio.gio_read(sfile2, sx))
    sy2 = np.squeeze(gio.gio_read(sfile2, sy))
    sz2 = np.squeeze(gio.gio_read(sfile2, sz))
    if(mode == 'halos'):
        sdid2 = np.squeeze(gio.gio_read(sfile2, 'desc_node_index'))  
        stag2 = np.squeeze(gio.gio_read(sfile2, 'fof_halo_tag'))
        smass2 = np.squeeze(gio.gio_read(sfile2, 'fof_halo_mass'))
    
    if(snapshotSubdirs): 
        sfiles3 = np.array(glob.glob('{}/STEP{}/*mpicosmo*'.format(spath, traj_steps[3])))
        sfiles3 = sfiles3[[not 'halo' in ss for ss in sfiles3]]
    else:
        sfiles3 = np.array(glob.glob('{}/*mpicosmo*.{}*'.format(spath, traj_steps[3])))
        sfiles3 = sfiles3[[not 'halo' in ss for ss in sfiles3]]
    if(not fullParticles): 
        sfiles3 = sfiles3[[not 'full' in ss for ss in sfiles3]]
    else:
        sfiles3 = sfiles3[['full' in ss for ss in sfiles3]]
    sfile3 = sorted(sfiles3)[0]
    print('reading snapshot at {}'.format(sfile3))
    sid3 = np.squeeze(gio.gio_read(sfile3, idName))
    sx3 = np.squeeze(gio.gio_read(sfile3, sx))
    sy3 = np.squeeze(gio.gio_read(sfile3, sy))
    sz3 = np.squeeze(gio.gio_read(sfile3, sz))
    if(mode == 'halos'):
        sdid3 = np.squeeze(gio.gio_read(sfile3, 'desc_node_index'))  
        stag3 = np.squeeze(gio.gio_read(sfile3, 'fof_halo_tag'))
    
    if(snapshotSubdirs): 
        sfiles4 = np.array(glob.glob('{}/STEP{}/*mpicosmo*'.format(spath, traj_steps[4])))
        sfiles4 = sfiles4[[not 'halo' in ss for ss in sfiles4]]
    else:
        sfiles4 = np.array(glob.glob('{}/*mpicosmo*.{}*'.format(spath, traj_steps[4])))
        sfiles4 = sfiles4[[not 'halo' in ss for ss in sfiles4]]
    if(not fullParticles): 
        sfiles4 = sfiles4[[not 'full' in ss for ss in sfiles4]]
    else:
        sfiles4 = sfiles4[['full' in ss for ss in sfiles4]]
    sfile4 = sorted(sfiles4)[0]
    print('reading snapshot at {}'.format(sfile4))
    sid4 = np.squeeze(gio.gio_read(sfile4, idName))
    sx4 = np.squeeze(gio.gio_read(sfile4, sx))
    sy4 = np.squeeze(gio.gio_read(sfile4, sy))
    sz4 = np.squeeze(gio.gio_read(sfile4, sz))
    if(mode == 'halos'):
        sdid4 = np.squeeze(gio.gio_read(sfile4, 'desc_node_index'))  
        stag4 = np.squeeze(gio.gio_read(sfile4, 'fof_halo_tag'))

    # loop through the ten particles selected for plotting, get their surrounding
    # snapshot data, and save the data as .npy files. The results can be plotting 
    # by calling plotLightconePaths() below
    for i in range(len(diffVals)):

        idx = diffVals[i]
                
        print('Matching to snapshots for idx {} with diff of {}'.format(idx,posDiff[idx]))
        print('Particle ID is {}'.format(iid[iMask][idx]))
        
        # lightcone data
        this_ix = ix[iMask][idx]
        this_iy = iy[iMask][idx]
        this_iz = iz[iMask][idx]
        this_ia = ia[iMask][idx]
        this_irot = irot[iMask][idx]
   
        this_ex = ex[eMask][idx]
        this_ey = ey[eMask][idx]
        this_ez = ez[eMask][idx]
        this_ea = ea[eMask][idx]
        this_erot = erot[eMask][idx]

        # make sure that box rotation for each particle above is the same 
        # (if not, we really screwed up somewhere)
        if(this_irot != this_erot):
            raise Exception('wuuuuuuut why has this happened')

        # find this particle/object in each snapshot
        if(mode == 'particles'):
            s0_idx = np.where(sid0 == iid[iMask][idx])
            s1_idx = np.where(sid1 == iid[iMask][idx])
            s2_idx = np.where(sid2 == iid[iMask][idx])
            s3_idx = np.where(sid3 == iid[iMask][idx])
            s4_idx = np.where(sid4 == iid[iMask][idx])
        
            sxi0 = sx0[s0_idx][0]
            sxi1 = sx1[s1_idx][0]
            sxi2 = sx2[s2_idx][0]
            sxi3 = sx3[s3_idx][0]
            sxi4 = sx4[s4_idx][0]
            
            syi0 = sy0[s0_idx][0]
            syi1 = sy1[s1_idx][0]
            syi2 = sy2[s2_idx][0]
            syi3 = sy3[s3_idx][0]
            syi4 = sy4[s4_idx][0]
            
            szi0 = sz0[s0_idx][0]
            szi1 = sz1[s1_idx][0]
            szi2 = sz2[s2_idx][0]
            szi3 = sz3[s3_idx][0]
            szi4 = sz4[s4_idx][0]
        
        if(mode == 'halos'):
            # match the halo tag from the lightcone to the merger tree
            # There will be several matches, corresponding to all of the
            # fragments associated with this halo in the merger tree. 
            # We can determine exactly which fragment we should be using by 
            # selecting the one with the largest mass, as was done in the
            # lightcone calculation
            # Recall that, for the halo lightcone, output step number 442
            # takes halo properties from snapshot *453*, because of backward
            # interpolation!
            if(iid[iMask][idx] < 0):
                s3_idx = np.where(stag3 == iid[iMask][idx])[0]
            else:
                s3_idx = np.where(stag3 == iid[iMask][idx])[0]

            # match the tree and descendent node idices to those
            # found above.
            s4_idx = np.where(sid4 == sdid3[s3_idx])[0]
           
            s2_idx = np.where(sdid2 == sid3[s3_idx])[0]
            if(len(s2_idx) > 1):
                fragMask = np.argmax(smass2[s2_idx])
                s2_idx = np.array([s2_idx[fragMask]])
            
            s1_idx = np.where(sdid1 == sid2[s2_idx])[0]
            if(len(s1_idx) > 1):
                fragMask = np.argmax(smass1[s1_idx])
                s1_idx = np.array([s1_idx[fragMask]])
            
            s0_idx = np.where(sdid0 == sid1[s1_idx])[0]
            if(len(s0_idx) > 1):
                fragMask = np.argmax(smass1[s0_idx])
                s0_idx = np.array([s0_idx[fragMask]])
        
            # now record snapshot positions of this halo, for which ever of
            # these four steps that it exists, starting at step 453 (it must exist
            # there, since that is the step on which our lightcone step 442 is based)
            sxi3 = sx3[s3_idx][0]
            syi3 = sy3[s3_idx][0]
            szi3 = sz3[s3_idx][0]
            
            if(sdid3[s3_idx] != -1):
                sxi4 = sx4[s4_idx][0]
                syi4 = sy4[s4_idx][0]
                szi4 = sz4[s4_idx][0]
            else:
                sxi4 = sxi3
                syi4 = syi3
                szi4 = szi3
            
            if(len(s2_idx) > 0):
                sxi2 = sx2[s2_idx][0]
                syi2 = sy2[s2_idx][0]
                szi2 = sz2[s2_idx][0]
            else:
                sxi2 = sxi3
                syi2 = syi3
                szi2 = szi3
            
            if(len(s1_idx) > 0):
                sxi1 = sx1[s1_idx][0]
                syi1 = sy1[s1_idx][0]
                szi1 = sz1[s1_idx][0]
            else:
                sxi1 = sxi2
                syi1 = syi2
                szi1 = szi2
            
            if(len(s0_idx) > 0):
                sxi0 = sx0[s0_idx][0]
                syi0 = sy0[s0_idx][0]
                szi0 = sz0[s0_idx][0]
            else:
                sxi0 = sxi1
                syi0 = syi1
                szi0 = szi1


        # get snapshot scale factors
        aa = np.linspace(1/(1+200), 1, 500)
        sai0 = aa[421]
        sai1 = aa[432]
        sai2 = aa[442]
        sai3 = aa[453]
        sai4 = aa[464]
        
        # true particles path for steps 421-464
        truex = np.array([sxi0, sxi1, sxi2, sxi3, sxi4])
        truey = np.array([syi0, syi1, syi2, syi3, syi4])
        truez = np.array([szi0, szi1, szi2, szi3, szi4])
        truea = np.array([sai0, sai1, sai2, sai3, sai4])
       
        if(mode == 'halos'):
            # approximated particle paths for steps 442-453 
            interpolx = np.array([this_ix, sxi3])
            interpoly = np.array([this_iy, syi3])
            interpolz = np.array([this_iz, szi3])
            interpola = np.array([this_ia, sai3])

            extrapx = np.array([this_ex, sxi3])
            extrapy = np.array([this_ey, syi3])
            extrapz = np.array([this_ez, szi3])
            extrapa = np.array([this_ea, sai3])

        elif(mode == 'particles'):
            # approximated particle paths for steps 442-453 
            if(solverMode == 'backward'):
                interpolx = np.array([this_ix, sxi3])
                interpoly = np.array([this_iy, syi3])
                interpolz = np.array([this_iz, szi3])
                interpola = np.array([this_ia, sai3])

                extrapx = np.array([this_ex, sxi3])
                extrapy = np.array([this_ey, syi3])
                extrapz = np.array([this_ez, szi3])
                extrapa = np.array([this_ea, sai3])
            
            elif(solverMode == 'forward'):
                interpolx = np.array([sxi2, this_ix])
                interpoly = np.array([syi2, this_iy])
                interpolz = np.array([szi2, this_iz])
                interpola = np.array([sai2, this_ia])

                extrapx = np.array([sxi2, this_ex])
                extrapy = np.array([syi2, this_ey])
                extrapz = np.array([szi2, this_ez])
                extrapa = np.array([sai2, this_ea])

       
        # done! save particle path data    
        np.save('{}/ix_{}.npy'.format(savePath, i), interpolx)
        np.save('{}/iy_{}.npy'.format(savePath, i), interpoly)
        np.save('{}/iz_{}.npy'.format(savePath, i), interpolz)
        np.save('{}/ia_{}.npy'.format(savePath, i), interpola)
        np.save('{}/iid_{}.npy'.format(savePath, i), iid[iMask][idx])
    
        np.save('{}/ex_{}.npy'.format(savePath, i), extrapx)
        np.save('{}/ey_{}.npy'.format(savePath, i), extrapy)
        np.save('{}/ez_{}.npy'.format(savePath, i), extrapz)
        np.save('{}/ea_{}.npy'.format(savePath, i), extrapa)

        np.save('{}/truex_{}.npy'.format(savePath, i), truex)
        np.save('{}/truey_{}.npy'.format(savePath, i), truey)
        np.save('{}/truez_{}.npy'.format(savePath, i), truez)
        np.save('{}/truea_{}.npy'.format(savePath, i), truea)
        print("saved {} for particle {}".format(diffRange, i))


#############################################################################################
#############################################################################################


def plotLightconePaths(dataPath, diffRange = 'max', plotMode='show', outDir='.'):
    '''
    This function plots the 3-dimensional path data as calculated and saved in 
    saveLightconePathData() above.

    Params:
    :param dataPath: Location of lightcone object path data (should match the outpath
                     argument of saveLightconePathData())
    :param diffRange: whether to use the 'max', 'med'(median) or 'min' diffVals (see 
                      doc strings in saveLightconePathData() for more info)
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless
                     otherwise specified in outDir.
    :param outDir: where to save figures, if plotMode == 'save'
    :return: None
    '''
    if plotMode not in ['show', 'save']:
        raise Exception('Unknown plotMode {}. Options are \'show\' or \'save\'.'.format(plotMode))
    if diffRange not in ['min', 'med', 'max']:
        raise Exception('Unknown diffRange {}. Options are \'min\', \'med\', or \'max\'.'
                        .format(diffRange))
    
    config(cmap=plt.cm.cool)

    # get data files
    data = '{}/{}_diff'.format(dataPath, diffRange)
    files = glob.glob('{}/truex_*'.format(data))
    fileNums = [int(f.split('_')[-1].split('.')[0]) for f in files]

    # loop through all files and plot (each file corresponds to one 
    # lightcone object)
    for i in fileNums:

        # read object trajectory data
        ix = np.load('{}/ix_{}.npy'.format(data, i))
        iy = np.load('{}/iy_{}.npy'.format(data, i))
        iz = np.load('{}/iz_{}.npy'.format(data, i))
        ia = np.load('{}/ia_{}.npy'.format(data, i))
        iid = np.load('{}/iid_{}.npy'.format(data, i))
    
        ex = np.load('{}/ex_{}.npy'.format(data, i))
        ey = np.load('{}/ey_{}.npy'.format(data, i))
        ez = np.load('{}/ez_{}.npy'.format(data, i))
        ea = np.load('{}/ea_{}.npy'.format(data, i))
        
        truex = np.load('{}/truex_{}.npy'.format(data, i))
        truey = np.load('{}/truey_{}.npy'.format(data, i))
        truez = np.load('{}/truez_{}.npy'.format(data, i))
        truea = np.load('{}/truea_{}.npy'.format(data, i))

        ax = plt.subplot2grid((3,3), (0,0), rowspan=2, colspan=2, projection='3d')
        x = np.random.randn(10)
        y = np.random.randn(10)
        z = np.random.randn(10)
        
        # ---------- main 3d plot ----------
        # plot true path
        ax.plot(truex, truey, truez, '--k.')
    
        # plot extrapolated and interpolated paths
        ax.plot(ex, ey, ez, '-o', lw=2)
        ax.plot(ix, iy, iz, '-o', lw=2)
        
        # plot star at starting point
        ax.plot([truex[0]], [truey[0]], [truez[0]], '*', ms=10)
        
        # formatting
        plt.title(r'$\mathrm{{ ID }}\>\>{}$'.format(iid), y=1.08, fontsize=18)
        ax.set_xlabel(r'$x\>\>\mathrm{(Mpc/h)}$', fontsize=12, labelpad=12)
        ax.set_ylabel(r'$y\>\>\mathrm{(Mpc/h)}$', fontsize=12, labelpad=12)
        ax.set_zlabel(r'$z\>\>\mathrm{(Mpc/h)}$', fontsize=12, labelpad=12)
        ax.tick_params(axis='both', which='major', labelsize=8)
        for t in range(len(ax.xaxis.get_major_ticks())): 
            if(t%2 == 1): 
                ax.xaxis.get_major_ticks()[t].label.set_color([1, 1, 1]) 
                ax.xaxis.get_major_ticks()[t].label.set_fontsize(0) 
        for t in range(len(ax.yaxis.get_major_ticks())): 
            if(t%2 == 1): 
                ax.yaxis.get_major_ticks()[t].label.set_color([1, 1, 1]) 
                ax.yaxis.get_major_ticks()[t].label.set_fontsize(0) 
        for t in range(len(ax.zaxis.get_major_ticks())): 
            if(t%2 == 1): 
                ax.zaxis.get_major_ticks()[t].label.set_color([1, 1, 1]) 
                ax.zaxis.get_major_ticks()[t].label.set_fontsize(0) 
        
        # ---------- subplot for x-a projection ----------
        ax_xa = plt.subplot2grid((3,3), (2,0), colspan=2)
        
        ax_xa.plot(truex, (1/truea)-1, '--k.')
        ax_xa.plot(ex, (1/ea)-1, '-o', lw=2)
        ax_xa.plot(ix, (1/ia)-1, '-o', lw=2)
        ax_xa.plot(truex[0], (1/truea[0])-1, '*', ms=10)
        
        # formatting
        ax_xa.set_xlabel(r'$x\>\>\mathrm{(Mpc/h)}$', fontsize=14, labelpad=6)
        ax_xa.set_ylabel(r'$\mathrm{redshift}$', fontsize=14, labelpad=6)
        ax_xa.set_yticks((1/truea)-1)
        ax_xa.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_xa.yaxis.tick_right()
        ax_xa.yaxis.set_label_position("right")
        ax_xa.invert_yaxis()
        ax_xa.grid()

        # ---------- subplot for z-a projection ----------
        ax_za = plt.subplot2grid((3,3), (0,2), rowspan=2)
        
        ax_za.plot((1/truea)-1, truez, '--k.', label='true dataPath')
        ax_za.plot((1/ea)-1, ez, '-o', lw=2, label = 'extrapolation')
        ax_za.plot((1/ia)-1, iz, '-o', lw=2, label='interpolation') 
        ax_za.plot((1/truea[0])-1, truez[0], '*', ms=10, label='starting position')
        
        # formatting 
        ax_za.set_ylabel(r'$z\>\>\mathrm{(Mpc/h)}$', fontsize=14, labelpad=6)
        ax_za.set_xlabel(r'$\mathrm{redshift}$', fontsize=14, labelpad=6)
        ax_za.set_xticks(1/(truea)-1)
        for tick in ax_za.get_xticklabels(): tick.set_rotation(90)
        ax_za.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_za.yaxis.tick_right()
        ax_za.yaxis.set_label_position("right")
        ax_za.grid()

        # legend
        ax_za.legend(bbox_to_anchor=(1.12, -0.35))

        # done
        plt.gcf().set_size_inches(8, 6)
        plt.gcf().tight_layout()
        plt.gcf().canvas.manager.window.move(540, 200)
        
        if(plotMode == 'show'):
            plt.show()
        else:
            plt.savefig('{}/lc_trajectory_{}Diff_{}'.format(outDir, diffRange, i))


#############################################################################################
#############################################################################################


def findDuplicates(lcDir, steps, lcSuffix, outDir, mode='particles', solverMode='forward', 
                   mergerTreeDir=None, mergerTreeSubdirs=False, 
                   initialVolumeOnly = False, rL=256):
    '''
    This function finds particle/object duplicates across timesteps in lightcone 
    output. Specifically, it finds particles that do not satisfy the following:
    The particle, defined by a tuple containing an id and box replication identifier,
    should be found only *once* in the lightcone, at one unique lightcone snapshot
    output file. If not, it means this particle has crossed the lightcone twice, 
    implying superluminal travel. It would then be considered a "duplicate particle".

    All duplicate particles that are found have the following data columns saved to 
    an hdf5 file: x, y, z, id. This data is saved for *both* of the particles in a 
    duplicate pair. Also, the total duplicate fraction relative to the whole population 
    is saved as an array of size 1.

    Params:
    :param lcDir: A directory containing lightcone output to search for particle
                  duplicates. It is assumed that this directory follows the naming 
                  and structure convention given in fig 6 of the Creating Lightcones 
                  in HACC notes (with one subdirectory per snapshot).
    :param steps: an array of at least two lightcone outputs, by snapshot number, between 
                  which to check for duplicate particles/objects. If mode == 'halos',
                  this array needs to contain three steps, the two that bound the 
                  lightcone timestep of interest, and the snapshot step following the
                  inital two ("following", as in, toward lower redshift). If the 
                  lightcone timestep of interest spans merger tree snapshot 475 and 487,
                  this array should be [475, 487, 499]
    :param lcSuffix: An identifier string that will be used as a filename suffix
                     for the output hdf5 files. This can be used to distinguish 
                     between multiple lightcones within which duplicates will be 
                     searched for.
    :param outDir: Location to save the output hdf5 file
    :param mode: whether to perform the object match-up on particles or
                 halos. If mode=="halos", then match on descendant merger 
                 tree indices rather than id
    :param solverMode: Wether the lightcone solver used to generate the output under 
                       validation was run in the standard forward mode (solverMode = 
                       'forward'), or the reversed mode, with extrapolation/interpolation
                       occuring backward in time (solverMode = 'backward')
    :param mergerTreeDir: If mode=='halos', then this argument must be passed as the path
                   to the top-level directory of the merger tree snapshots that were 
                   used to run the lightcone
    :param mergerTreeSubdirs: If true, assume that the merger tree data is grouped into
                              subdirectories of the format "STEPXXX". Otherwise, 
                              assume one flat directory with the the step number
                              "XXX" in the filenames somewhere, as "*.XXX.*" where
                              * is a wildcard
    :param initialVolumeOnly: Whether or not to omit all simulation box replications in the
                              duplication check. If False, objects will be matched by both 
                              their id and their replication identifier. If not, only the id
                              will be used for the matchup, andall other box replications will
                              be omitted. Turn this option on to debug the validation test 
                              itself (the number of deuplications found in the initial volume
                              should of course be the same in either case)
    :param rL: The simulation box length in Mpc/h. This value only used if initialVolumeOnly==`True
    :return: None
    '''

    if(len(steps) != 2 and mode == 'particles'):
        raise Exception('Exactly two step numbers should be passed in the \'steps\' arg if '\
                        'mode==\'particles\'')
    if(len(steps) != 3 and steps[-1] != 499 and mode == 'halos'):
        raise Exception('Exactly three step numbers should be passed in the \'steps\' arg if '\
                        'mode==\'halos\'')
    if mode not in ['particles', 'halos']:
        raise Exception('Unknown mode {}. Options are \'particles\' or \'halos\'.'.format(mode))
    if(mode == 'halos' and mergerTreeDir == None):
        raise Exception('if mode==\'halos\', fofDir must be passed')

    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio
    
    subdirs = glob.glob('{}/*'.format(lcDir))
    steps = sorted(steps)
    
    # get lc subdirectory prefix (could be 'lc' or 'lcGals', etc.). 
    # prefix of subdirs in epath and lcDir assumed to be the same.
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue
    
    # create output hdf5 file
    print('\nCreating output hdf5 for {} step {}'.format(lcSuffix, steps[0]))
    outfile = h5.File('{}/duplicates_{}_{}.hdf5'.format(outDir, lcSuffix, steps[0]), 'w')

    # if the timestep passed is 487 - 499, write out immediately, since there
    # cannot possibly be duplicates
    if(steps[0] == 487):
        
        file1 = sorted(glob.glob('{}/{}{}/*'.format(lcDir, prefix, steps[0])))[0]
        ids1 = np.squeeze(gio.gio_read(file1, 'id'))
        total1 = len(ids1)

        print('repeat fraction is 0.0')
        print('writing out {}'.format('{}/duplicates_{}_{}.hdf5'.format(outDir, lcSuffix, steps[0])))
        outfile.create_dataset('repeat_frac', data=np.array([]))
        outfile.create_dataset('total_count_step{}'.format(steps[0]), data=np.array([total1]))
        outfile.create_dataset('total_count_step{}'.format(steps[1]), data=np.array([0]))
        outfile.create_dataset('id_step{}'.format(steps[0]), data = np.array([]))
        outfile.create_dataset('id_step{}'.format(steps[1]), data = np.array([]))
        outfile.create_dataset('replication_step{}'.format(steps[0]), np.array([]))
        outfile.create_dataset('replication_step{}'.format(steps[1]), data = np.array([]))
        outfile.create_dataset('x_step{}'.format(steps[0]), data = np.array([]))
        outfile.create_dataset('x_step{}'.format(steps[1]), data = np.array([]))
        outfile.create_dataset('y_step{}'.format(steps[0]), data = np.array([]))
        outfile.create_dataset('y_step{}'.format(steps[1]), data = np.array([]))
        outfile.create_dataset('z_step{}'.format(steps[0]), data = np.array([]))
        outfile.create_dataset('z_step{}'.format(steps[1]), data = np.array([]))
        outfile.create_dataset('vx_step{}'.format(steps[0]), data = np.array([]))
        outfile.create_dataset('vx_step{}'.format(steps[1]), data = np.array([]))
        outfile.create_dataset('vy_step{}'.format(steps[0]), data = np.array([]))
        outfile.create_dataset('vy_step{}'.format(steps[1]), data = np.array([]))
        outfile.create_dataset('vz_step{}'.format(steps[0]), data = np.array([]))
        outfile.create_dataset('vz_step{}'.format(steps[1]), data = np.array([]))
        outfile.create_dataset('a_step{}'.format(steps[0]), data = np.array([]))
        outfile.create_dataset('a_step{}'.format(steps[1]), data = np.array([]))
        return


    # sort the subdirectory contents and take the first item, since that
    # should always be the GIO header file
    print('reading data')
    file1 = sorted(glob.glob('{}/{}{}/*'.format(lcDir, prefix, steps[0])))[0]
    file2 = sorted(glob.glob('{}/{}{}/*'.format(lcDir, prefix, steps[1])))[0]

    if(mode == 'particles'): 
        # read particle data
        ids1 = np.squeeze(gio.gio_read(file1, 'id'))
        repl1 = np.squeeze(gio.gio_read(file1, 'replication'))
        ids2 = np.squeeze(gio.gio_read(file2, 'id'))
        repl2 = np.squeeze(gio.gio_read(file2, 'replication'))
    
    elif(mode == 'halos'):
        # read halo data
        ids1 = np.squeeze(gio.gio_read(file1, 'id'))
        repl1 = np.squeeze(gio.gio_read(file1, 'replication'))
        ids2 = np.squeeze(gio.gio_read(file2, 'id'))
        repl2 = np.squeeze(gio.gio_read(file2, 'replication'))
        
        # open merger tree files to do matchup
        if(mergerTreeSubdirs):
            if(solverMode == 'backward'):
                mt_file1 = sorted(glob.glob('{}/STEP{}/*'.format(mergerTreeDir, steps[1])))[0]
                mt_file2 = sorted(glob.glob('{}/STEP{}/*'.format(mergerTreeDir, steps[2])))[0]
            else:    
                mt_file1 = sorted(glob.glob('{}/STEP{}/*'.format(mergerTreeDir, steps[0])))[0]
                mt_file2 = sorted(glob.glob('{}/STEP{}/*'.format(mergerTreeDir, steps[1])))[0]
        else:
            if(solverMode == 'backward'):
                mt_file1 = sorted(glob.glob('{}/*.{}.*'.format(mergerTreeDir, steps[1])))[0]
                mt_file2 = sorted(glob.glob('{}/*.{}.*'.format(mergerTreeDir, steps[2])))[0]
            else:
                mt_file1 = sorted(glob.glob('{}/*.{}.*'.format(mergerTreeDir, steps[0])))[0]
                mt_file2 = sorted(glob.glob('{}/*.{}.*'.format(mergerTreeDir, steps[1])))[0]

        
        mt_ids1 = np.squeeze(gio.gio_read(mt_file1, 'fof_halo_tag'))
        mt_desc_indices1 = np.squeeze(gio.gio_read(mt_file1, 'desc_node_index'))
        mt_tree_indices2 = np.squeeze(gio.gio_read(mt_file2, 'tree_node_index'))
        mt_ids2 = np.squeeze(gio.gio_read(mt_file2, 'fof_halo_tag'))
        if(solverMode == 'forward'):
            # if we ran the halos forward, then we must have been extrapolating, 
            # in which case we must have been using an fof catalog as input, so let's
            # mask the merger tree tags to recover the fof tags there
            mt_ids1 = (np.sign(mt_ids1)) * mt_ids1 & 0xffffffffffff
            mt_ids2 = (np.sign(mt_ids2)) * mt_ids2 & 0xffffffffffff

        # find location of each lightcone object in earlier step in merger tree
        lc1_to_mt1 = search_sorted(mt_ids1, ids1)
        if(np.sum(lc1_to_mt1==-1) != 0): 
            raise Exception('output in lightcone file {} not found in merger trees, '\
                            'maybe passed the wrong files?'.format(file1))
        
        # get descendent node indices of every halo in earlier lc step
        desc_ids1 = mt_desc_indices1[lc1_to_mt1]
        
        # find location of each lightcone object in later step in merger tree
        lc2_to_mt2 = search_sorted(mt_ids2, ids2)
        if(np.sum(lc2_to_mt2==-1) != 0): 
            raise Exception('output in lightcone file {} not found in merger trees, '\
                            'maybe passed the wrong files?'.format(file2))
        
        # get tree node indices of every halo in later lc step
        tree_ids2 = mt_tree_indices2[lc2_to_mt2]
        
        # ok got everything we need. We want to compare the desc node indices
        # at lc step 1 with the tree node indices at lc step 2 (step 2 is later)
        ids1 = desc_ids1
        ids2 = tree_ids2
         
    if(initialVolumeOnly):
        # get rid of everything not in the initial volume (don't
        # consider objects found in replicated boxes, so matchup
        # is simplified)
        print('Removing objects outside of initial simulation volume')
        x1 = np.squeeze(gio.gio_read(file1, 'x'))
        y1 = np.squeeze(gio.gio_read(file1, 'y'))
        z1 = np.squeeze(gio.gio_read(file1, 'z'))
        x2 = np.squeeze(gio.gio_read(file2, 'x'))
        y2 = np.squeeze(gio.gio_read(file2, 'y'))
        z2 = np.squeeze(gio.gio_read(file2, 'z'))
        
        initVolMask1 = np.logical_and.reduce((abs(x1) < rL, 
                                              abs(y1) < rL, 
                                              abs(z1) < rL))
         
        initVolMask2 = np.logical_and.reduce((abs(x2) < rL, 
                                              abs(y2) < rL, 
                                              abs(z2) < rL))

        # get rid of any particles that do not originate from the inital
        # volume, but had their positions enter the volume through the
        # lightcone position extrapolation
        boxesFound1 = np.unique(repl1[initVolMask1], return_counts=True)
        initVol1 = boxesFound1[0][np.argmax(boxesFound1[-1])]
        boxMask1 = repl1 == initVol1
        
        print('initial volume at output {}: {}'.format(steps[0], initVol1))
        
        ids1 = ids1[boxMask1]
        repl1 = repl1[boxMask1]
        
        boxesFound2 = np.unique(repl2[initVolMask2], return_counts=True)
        initVol2 = boxesFound2[0][np.argmax(boxesFound2[-1])]
        boxMask2 = repl2 == initVol2
        print('initial volume at output {}: {}'.format(steps[1], initVol2))
        
        ids2 = ids2[boxMask2]
        repl2 = repl2[boxMask2]
    
    else:
        boxMask1 = np.ones(len(repl1), dtype=bool)
        boxMask2 = np.ones(len(repl2), dtype=bool)
    
    total1 = len(ids1)
    total2 = len(ids2)

    # do matchup
    print('matching')
    idMatches = search_sorted(ids1, ids2)
    
    # remove matches that aren't in the same box replication
    for i in range(len(idMatches)):
        if(idMatches[i] == -1): continue
        j = idMatches[i]
        if(repl2[i] != repl1[j]):
            idMatches[i] = -1
    
    matchesMask2 = idMatches != -1
    matchesMask1 = idMatches[matchesMask2]
    
    print('found {} duplicates'.format(np.sum(matchesMask2)))
    
    # get data for all duplicate objects
    repl1 = repl1[matchesMask1]
    x1 = np.squeeze(gio.gio_read(file1, 'x')[boxMask1][matchesMask1])
    y1 = np.squeeze(gio.gio_read(file1, 'y')[boxMask1][matchesMask1])
    z1 = np.squeeze(gio.gio_read(file1, 'z')[boxMask1][matchesMask1])
    vx1 = np.squeeze(gio.gio_read(file1, 'vx')[boxMask1][matchesMask1])
    vy1 = np.squeeze(gio.gio_read(file1, 'vy')[boxMask1][matchesMask1])
    vz1 = np.squeeze(gio.gio_read(file1, 'vz')[boxMask1][matchesMask1])
    a1 = np.squeeze(gio.gio_read(file1, 'a')[boxMask1][matchesMask1])
    dup_ids1 = np.squeeze(gio.gio_read(file1, 'id')[boxMask1][matchesMask1])
    
    repl2 = repl2[matchesMask2]
    x2 = np.squeeze(gio.gio_read(file2, 'x')[boxMask2][matchesMask2])
    y2 = np.squeeze(gio.gio_read(file2, 'y')[boxMask2][matchesMask2])
    z2 = np.squeeze(gio.gio_read(file2, 'z')[boxMask2][matchesMask2])
    vx2 = np.squeeze(gio.gio_read(file2, 'vx')[boxMask2][matchesMask2])
    vy2 = np.squeeze(gio.gio_read(file2, 'vy')[boxMask2][matchesMask2])
    vz2 = np.squeeze(gio.gio_read(file2, 'vz')[boxMask2][matchesMask2])
    a2 = np.squeeze(gio.gio_read(file2, 'a')[boxMask2][matchesMask2])
    dup_ids2 = np.squeeze(gio.gio_read(file2, 'id')[boxMask2][matchesMask2])

    if( np.sum(abs(dup_ids1 - dup_ids2)) != 0 and np.sum(abs(repl1 - repl2)) != 0 ): 
        raise Exception('non-duplicates marked as duplicates')
    
    repeat_frac = float(len(dup_ids1)) / len(ids1) 
    print('repeat fraction is {}'.format(repeat_frac))

    print('writing out {}'.format('{}/duplicates_{}_{}.hdf5'.format(outDir, lcSuffix, steps[0])))
    outfile.create_dataset('repeat_frac', data=np.array([repeat_frac]))
    outfile.create_dataset('total_count_step{}'.format(steps[0]), data=np.array([total1]))
    outfile.create_dataset('total_count_step{}'.format(steps[1]), data=np.array([total2]))
    outfile.create_dataset('id_step{}'.format(steps[0]), data = dup_ids1)
    outfile.create_dataset('id_step{}'.format(steps[1]), data = dup_ids2)
    outfile.create_dataset('replication_step{}'.format(steps[0]), data = repl1)
    outfile.create_dataset('replication_step{}'.format(steps[1]), data = repl2)
    outfile.create_dataset('x_step{}'.format(steps[0]), data = x1)
    outfile.create_dataset('x_step{}'.format(steps[1]), data = x2)
    outfile.create_dataset('y_step{}'.format(steps[0]), data = y1)
    outfile.create_dataset('y_step{}'.format(steps[1]), data = y2)
    outfile.create_dataset('z_step{}'.format(steps[0]), data = z1)
    outfile.create_dataset('z_step{}'.format(steps[1]), data = z2)
    outfile.create_dataset('vx_step{}'.format(steps[0]), data = vx1)
    outfile.create_dataset('vx_step{}'.format(steps[1]), data = vx2)
    outfile.create_dataset('vy_step{}'.format(steps[0]), data = vy1)
    outfile.create_dataset('vy_step{}'.format(steps[1]), data = vy2)
    outfile.create_dataset('vz_step{}'.format(steps[0]), data = vz1)
    outfile.create_dataset('vz_step{}'.format(steps[1]), data = vz2)
    outfile.create_dataset('a_step{}'.format(steps[0]), data = a1)
    outfile.create_dataset('a_step{}'.format(steps[1]), data = a2)


#############################################################################################
#############################################################################################


def compareDuplicates(duplicatePath, steps, lcSuffix, compType='scatter', plotMode='show', outDir='.'):
    '''
    This function visualizes the output of the validation function, 
    findDuplicates(). That function finds all particle duplicates between
    two lightcone steps (explained in detail in the findDuplicates() doc
    string) and outputs their positions to an hdf5 file. This function 
    opens that file, and plots a downsampling of those particles on 3d
    axes.

    Two runs of findDuplicates() is expected to have been executed, with results 
    saved to the same location. This is becuase the intended use of this 
    validation test is to inspect the duplicates present in a lightcone
    timestep resultant from running the old (extrapolation) vs new (interpolation)
    methods. 

    Params:
    :param deplicatePath: the output directory of two previous runs of 
                          findDuplicates().
    :param steps: an array of at least two lightcone outputs, by snapshot number, between 
                  which a corresponding run(s) of finDuplicates() checked for duplicate 
                  particles/objects. If len(steps) == 2, the comparison plot will be made
                  as a 3d scatter of all duplicate particle positions (compType='scatter'). 
                  If len(steps) > 2, the comparison plot will be made as a histogram of 
                  duplicate counts vs redshift (compType='hist').
    :param lcSuffix: An identifier string that will be assumed as a filename suffix
                     for the input hdf5 files. This should be array of length 2, with
                     each element agreeing with the lcSuffix input for a run of 
                     findDuplicates() which wrote to the directory duplicatePath
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless otherwise
                     specified in the 'outDir' arg
    :param outDir: where to save figures, if plotMode == 'save'
    :return: None
    '''
    if plotMode not in ['show', 'save']:
        raise Exception('Unknown plotMode {}. Options are \'show\' or \'save\'.'.format(plotMode))
    if(len(lcSuffix) != 2):
        raise Exception('Input array \'lcSuffix\' should be of length 2. See function docstrings')
    if(len(steps)%2 != 0):
        raise Exception('Length of input array \'steps\' should be divisible by 2. See function docstrings')

    # ---------- scatter plot (single timestep) comparison ----------
    # ---------------------------------------------------------------
    if(len(steps) == 2):
        
        # open files
        dupl1 = h5.File('{}/duplicates_{}_{}.hdf5'.format(duplicatePath, lcSuffix[0], steps[0]), 'r')
        dupl2 = h5.File('{}/duplicates_{}_{}.hdf5'.format(duplicatePath, lcSuffix[1], steps[0]), 'r')

        print('Duplicate fraction for {} output: {}'.format(lcSuffix[0], dupl1['repeat_frac'][:][0]))
        print('Duplicate fraction for {} output: {}'.format(lcSuffix[1], dupl2['repeat_frac'][:][0]))
        
        # setup plotting
        f = plt.figure(plt.gcf().number+1, figsize=(12, 6))
        axe = f.add_subplot(121, projection='3d')
        axi = f.add_subplot(122, projection='3d')
        title = f.suptitle('step {} - step {}'.format(steps[0], steps[-1]))
        axi.set_title('{}\nDuplicate fraction: {:.4f}%'.format(lcSuffix[0], dupl1['repeat_frac'][:][0]*100))
        axe.set_title('{}\nDuplicate fraction: {:.4f}%'.format(lcSuffix[1], dupl2['repeat_frac'][:][0]*100))

        # find intersection and symmetric difference of the two outputs
        maski = np.in1d(dupl1['id_step{}'.format(steps[0])], dupl2['id_step{}'.format(steps[0])])
        maske = np.in1d(dupl2['id_step{}'.format(steps[0])], dupl1['id_step{}'.format(steps[0])])

        # downsample extrapolated output for faster plotting
        if(len(maske) > 1000):
            maske_nokeep = np.random.choice(np.where(~maske)[0], 
                                            int(len(np.where(~maske)[0])*0.9), replace=False)
            maske[maske_nokeep] = 1
            e_downsample_idx = np.random.choice(
                               np.linspace(0, len(dupl2['id_step{}'.format(steps[0])][:])-1, 
                                           len(dupl2['id_step{}'.format(steps[0])][:]), dtype=int), 
                               int(len(dupl2['id_step{}'.format(steps[0])][:])*0.1), 
                               replace=False)

            e_downsample = np.zeros(len(dupl2['id_step{}'.format(steps[0])][:]), dtype = bool)
            e_downsample[e_downsample_idx] = 1
        
        else:
            e_downsample = np.ones(len(dupl2['id_step{}'.format(steps[0])][:]), dtype=bool)
        
        # do plotting. The extrapolation is downsampled, while the interpolated output
        # is not, since the extrapolated output should have far mroe duplicate objects
        
        axe.plot(dupl2['x_step{}'.format(steps[0])][e_downsample], 
                 dupl2['y_step{}'.format(steps[0])][e_downsample], 
                 dupl2['z_step{}'.format(steps[0])][e_downsample], 
                '.g', ms=1, label='shared duplicates')

        axe.plot(dupl2['x_step{}'.format(steps[0])][~maske], 
                 dupl2['y_step{}'.format(steps[0])][~maske], 
                 dupl2['z_step{}'.format(steps[0])][~maske], 
                 '+m', mew=1, label='unique duplicates')
        
        axe.set_xlabel('x (Mpc/h)')
        axe.set_ylabel('y (Mpc/h)')
        axe.set_zlabel('y (Mpc/h)')
        axe.legend(bbox_to_anchor=(0.1, 0.1))

        axi.plot(dupl1['x_step{}'.format(steps[0])], 
                 dupl1['y_step{}'.format(steps[0])], 
                 dupl1['z_step{}'.format(steps[0])], 
                 '.b', ms=1, label='shared duplicates')

        axi.plot(dupl1['x_step{}'.format(steps[0])][~maski], 
                dupl1['y_step{}'.format(steps[0])][~maski], 
                dupl1['z_step{}'.format(steps[0])][~maski], 
                 '+r', mew=1, label='unique duplicates')
        
        axi.set_xlabel('x (Mpc/h)')
        axi.set_ylabel('y (Mpc/h)')
        axi.set_zlabel('y (Mpc/h)')
        axi.legend(bbox_to_anchor=(0.1, 0.1))
    
    # ---------- histogram (multi-timestep) comparison ----------
    # -----------------------------------------------------------
    else:
        
        # setup plotting
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, 6))
        f = plt.figure(plt.gcf().number+1)
        ax = f.add_axes((0.1, 0.25, 0.8, 0.65))
        ax.set_title('step {} - step {}'.format(steps[0], steps[-1]))
        ax.set_ylabel('cdf')
        ax.set_xlabel('redshift')

        # add second set of x-axis labels
        ax2 = f.add_axes((0.1, 0.1, 0.8, 0.0))
        ax2.yaxis.set_visible(False)
        ax2.set_xlabel('step')

        dupNum1 = np.zeros(len(steps))
        dupNum2 = np.zeros(len(steps))
        total1 = np.zeros(len(steps))
        total2 = np.zeros(len(steps))
        
        # loop over all input steps
        steps = sorted(steps)
        for k in range(len(steps)-1):

            if(steps[k] == 487):
                dupl1 = h5.File('{}/duplicates_{}_{}.hdf5'.format(duplicatePath, lcSuffix[0], steps[k]), 'r')
                total1[k] = dupl1['total_count_step{}'.format(steps[k])][:][0]
                dupNum1[k] = np.cumsum(total1[::-1])[-1] * 1e-7
                dupNum2[k] = np.cumsum(total2[::-1])[-1] * 1e-7
                continue
            
            dupl1 = h5.File('{}/duplicates_{}_{}.hdf5'.format(duplicatePath, lcSuffix[0], steps[k]), 'r')
            dupl2 = h5.File('{}/duplicates_{}_{}.hdf5'.format(duplicatePath, lcSuffix[1], steps[k]), 'r')
            total1[k] = dupl1['total_count_step{}'.format(steps[k])][:][0]
            total2[k] = dupl2['total_count_step{}'.format(steps[k])][:][0]
            a1 = dupl1['a_step{}'.format(steps[k])][:]
            a2 = dupl2['a_step{}'.format(steps[k])][:]
            dupNum1[k] = len(a1)
            dupNum2[k] = len(a2)

        a = np.linspace(1/(200+1), 1, 500)
        step_zs = (1/a[steps]) - 1
        
        ax.plot(step_zs[::-1], 
                 np.cumsum(dupNum1[::-1]) / np.cumsum(total1[::-1])[-1], '-s', color=colors[1], lw=1.2, ms=5)
        ax.plot(step_zs[::-1], 
                 np.cumsum(dupNum2[::-1]) / np.cumsum(total2[::-1])[-1], '-s', color=colors[-2], lw=1.2, ms=5)
        
        ax.set_yscale('log', nonposy='clip')
        ax.set_ylim([0.0000001,1.0])
        ax.set_xlim([min(step_zs), max(step_zs)])
        ax.set_xticks(step_zs)
        ax.grid()
        
        ax2.set_xlim([min(step_zs), max(step_zs)])
        ax2.set_xticks(step_zs)
        ax2.set_xticklabels(steps)
        
    if(plotMode == 'show'):
        plt.show()
    else:
        f.savefig('{}/lc_duplicates_{}-{}'.format(outDir, steps[0], steps[1]))


#############################################################################################
#############################################################################################


def compareReps(lcDir1, lcDir2, step, skyArea='octant', plotMode='show', outDir='.'):
    '''
    This function compares the simulation box replication rotations
    between two lightcones. The intended reason for running this validation
    test is to ensure that two lightcone runs given the same random seed
    produce the same box rotations. It will plot all box replications present 
    in the output at the input step in 3d space, with each box being color
    coded by its rotation value. Refer to the Creating Lightcones in HACC 
    document to see the details of box rotation and replication

    Params:
    :param lcDir1: A directory containing lightcone output. 
                   It is assumed that this directory follows the 
                   naming and structure convention given in fig 6 of 
                   the Creating Lightcones in HACC notes (with one 
                   subdirectory per snapshot).
    :param lcDir2: A second directory containing lightcone output, 
                   following the description of lcDir1, to compare to 
                   the output of lcDir1.
    :param step: The step to use in the comparison. Different steps 
                 will generally plot box replications at different 
                 spatial positions (those that overlap the shell of the
                 lightcone at the step)
    :param skyArea: Whether the input lightcone fills on octant (skyArea = 'octant')
                    of the sky, or the full sky (skyArea = 'full')
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless otherwise
                     specified in the 'outDir' arg
    :param outDir: where to save figures, if plotMode == 'save'
    :return: None
    '''
    if plotMode not in ['show', 'save']:
        raise Exception('Unknown plotMode {}. Options are \'show\' or \'save\'.'.format(plotMode))
    if skyArea not in ['octant', 'full']:
        raise Exception('Unknown skyArea arg {}. Options are \'octant\' or \'full\'.'.format(skyArea))
    
    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio
   
    # setup plottong
    f = plt.figure(plt.gcf().number+1)
    ax1 = f.add_subplot(221, projection='3d')
    ax2 = f.add_subplot(223, projection='3d')
    
    # find subdir prefix 
    subdirs = glob.glob('{}/*'.format(lcDir1)) 
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue

    # sort the subdirectory contents and take the first item, since that
    # should always be the GIO header file
    file1 = sorted(glob.glob("{}/{}{}/*".format(lcDir1, prefix, step)))[0]
    file2 = sorted(glob.glob("{}/{}{}/*".format(lcDir2, prefix, step)))[0]

    rot1 = gio.gio_read(file1, 'rotation')
    rep1 = gio.gio_read(file1, 'replication')
    rot2 = gio.gio_read(file2, 'rotation')
    rep2 = gio.gio_read(file2, 'replication')
    
    if(skyArea == 'octant'):
        LCReplicants1 = (len(np.unique(rep1)) ** (1/3)) - 1
        LCReplicants2 = (len(np.unique(rep2)) ** (1/3)) - 1
    elif(skyArea == 'full'):
        LCReplicants1 = ((len(np.unique(rep1)) ** (1/3)) - 2) / 2
        LCReplicants2 = ((len(np.unique(rep2)) ** (1/3)) - 2) / 2 

    # find all box replications
    uniqRep1 = sorted(np.unique(rep1))
    uniqRep2 = sorted(np.unique(rep2))
    
    # sample color map for each of the 6 possible rotation values, as described in 
    # section 4.5 the Creating Lightcones in HACC doc
    colors = plt.cm.viridis(np.linspace(0, 1, 6))

    for j in range(len(uniqRep1)):
        
        # recover box replication spatial positions via the perscription described in 
        # section 4.5 of the Creating Lightcones in HACC doc
        xReps1 = LCReplicants1 - (uniqRep1[j] >> 20)
        yReps1 = LCReplicants1 - ((uniqRep1[j] >> 10) & 0x3ff)
        zReps1 = LCReplicants1 - (uniqRep1[j] & 0x3ff)
        this_rot1 = np.squeeze(rot1[np.where(rep1 == uniqRep1[j])[0]])
        if(np.sum(abs(np.diff(this_rot1))) != 0): 
            print('Particles in lightcone output {} in the same box replication have different'\
                  'rotations... you seriously messed something up badly'.format(lcDir1))
            return

        xReps2 = LCReplicants2 - (uniqRep2[j] >> 20)
        yReps2 = LCReplicants2 - ((uniqRep2[j] >> 10) & 0x3ff)
        zReps2 = LCReplicants2 - (uniqRep2[j] & 0x3ff)
        this_rot2 = np.squeeze(rot2[np.where(rep2 == uniqRep2[j])[0]])
        if(np.sum(abs(np.diff(this_rot2))) != 0): 
            print('Particles in lightcone output {} in the same box replication have different'\
                  'rotations... you seriously messed something up badly'.format(lcDir2))
            return

        # plot
        plotBox(xReps1, yReps1, zReps1, 1, 1, 1, ax1, colors[this_rot1[0]])
        plotBox(xReps2, yReps2, zReps2, 1, 1, 1, ax2, colors[this_rot2[0]])

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    # make legend of possible rotations 
    ax1.plot([0,0], [0,0], '-', lw=10, color=colors[0], label='no rotations')
    ax1.plot([0,0], [0,0], '-', lw=10, color=colors[1], label='x <--> y')
    ax1.plot([0,0], [0,0], '-', lw=10, color=colors[2], label='y <--> z')
    ax1.plot([0,0], [0,0], '-', lw=10, color=colors[3], label='x <--> y and\ny <--> z')
    ax1.plot([0,0], [0,0], '-', lw=10, color=colors[4], label='z <--> x')
    ax1.plot([0,0], [0,0], '-', lw=10, color=colors[5], label='x <--> y and\nz <--> x')
    ax1.legend(bbox_to_anchor=(2, 0.8))

    ax1.set_title('lcDir1')
    ax2.set_title('lcDir2')
    
    if(plotMode == 'show'):
        plt.show()
    else:
        f.savefig('{}/lc_boxRotations_{}'.format(outDir, step))


#############################################################################################
#############################################################################################


def comvDist_vs_z(lcDirs, steps, lcNames=['uncorrected', 'second order corrections w/ weighting'], 
                  twoPanel=True, cosmology='alphaQ', plotMode='show', outDir='.'):
    '''
    This function computes and plots the error in the comoving-distance 
    vs. redshift relation for lightcone outputs, using that returned by 
    astropy, given an AlphaQ cosmology, as truth. This is to visualize
    discreteness effects in the relation arising from discretized simulation
    outputs, and the approximations present in our lightcone solver code.

    Params:
    :param steps: A list of steps to include in the generated plot
    :param lcDirs: A list of directories containing lightcone output. 
                   It is assumed that these directories follow the 
                   naming and structure convention given in fig 6 of 
                   the Creating Lightcones in HACC notes (with one 
                   subdirectory per snapshot). 
    :param lcNames: A list of strings to represent the lightcone outputs
                   given in lcDirs. This is for plotting comparisons, 
                   to fill in the legend labels. The length of this array 
                   should of course match that of lcDirs
    :param twoPanel: If true, then the generated figure will contain two
                     axes vertically arranged. The top plot will be the
                     comving distance vs. redshift relation, with the 
                     lightcone output and astropy "truth" plotted. The
                     bottom plot is the same, but the y-axis is the error, 
                     or difference between the lc output and the truth. If
                     false, only generate the error (bottom) axes.
    :param cosmology: simulation from which to assume cosmological parameters.
                      Default is 'alphaQ'
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless otherwise
                     specified in the 'outDir' arg
    :param outDir: where to save figures, if plotMode == 'save'
    :return: None
    '''
    if plotMode not in ['show', 'save']:
        raise Exception('Unknown plotMode {}. Options are \'show\' or \'save\'.'.format(plotMode))
    
    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio

    # get lc subdirectory prefix of each input lightcone
    # (prefix could be 'lc' or 'lcGals', etc.). 
    prefix = ['']*len(lcDirs)
    for j in range(len(prefix)):
        subdirs = glob.glob('{}/*'.format(lcDirs[j]))
        for i in range(len(subdirs[0].split('/')[-1])):
            try:
                (int(subdirs[0].split('/')[-1][i]))
                prefix[j] = subdirs[0].split('/')[-1][0:i]
                break
            except ValueError:
                continue
    
    # define cosmology
    if(cosmology=='alphaQ'):
        cosmo = FlatLambdaCDM(H0=71, Om0=0.26479, Ob0=0.044792699861138666, Tcmb0 = 0, Neff = 0)
    else:
        raise Exception('Unknown cosmology, {}. Available is: alphaQ'.format(cosmology))
    
    # get step redshifts
    a = np.linspace(1./(200.+1.), 1., 500)
    z = 1./a-1.
    zs = z[steps]
    
    # set up plotting
    config(cmap=plt.cm.plasma, numColors=3)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    f = plt.figure(plt.gcf().number+1)
    
    if(twoPanel):
        ax1 = f.add_subplot(211)
        ax2 = f.add_subplot(212, sharex=ax1)
        ax1.set_title('z = 0 -> {:.2f}'.format(max(zs)))
    else:
        ax2 = f.add_subplot(111)
        ax2.set_title('z = 0 -> {:.2f}'.format(max(zs)))

    # define arrays to be plotted after looping through all step data.
    # these will end up being arrays of array objects (NOT multidimensional
    # arrays, as the different lc outputs given by lcDirs probably have
    # different particle/object counts)
    r_all = [np.array([])]*len(lcDirs)
    r_true_all = [np.array([])]*len(lcDirs)
    a_all = [np.array([])]*len(lcDirs)

    # start looping through arrays
    for n in range(len(lcDirs)):
        lcDir = lcDirs[n]
        print('working on lc at {}'.format(lcDir))
        
        for step in steps:
            print('working on step {}'.format(step))
            
            # sort all files in this directory to get the gio header file
            lc = sorted(glob.glob('{0}/{1}{2}/*'.format(lcDir, prefix[n], step)))[0]
            
            # read and subsample
            x = gio.gio_read(lc, 'x')
            y = gio.gio_read(lc, 'y')
            z = gio.gio_read(lc, 'z')
            a = gio.gio_read(lc, 'a')
            
            mask = np.random.choice(np.arange(len(x)),10000)
            mask = mask[np.argsort(np.ndarray.flatten(a[mask]))]
            mask = np.ndarray.flatten(mask)
            x = x[mask]
            y = y[mask]
            z = z[mask]
            a = a[mask]
            r = np.sqrt(x**2. + y**2. + z**2.)
            r_true = cosmo.comoving_distance(1/a-1).value*cosmo.h

            # add data to _all arrays
            a_all[n] = np.hstack([a_all[n], np.ndarray.flatten(a)[::-1]])
            r_all[n] = np.hstack([r_all[n], np.ndarray.flatten(r)[::-1]]) 
            r_true_all[n] = np.hstack([r_true_all[n], np.ndarray.flatten(r_true)[::-1]])
    
        # plot comv dist vs scale factor
        if(twoPanel):
            ax1.plot(a_all[n], r_all[n], color=colors[2*n], lw=1, label=lcNames[n])
            
            # plot the truth only once, even though it had to be computed for each 
            # lightcone output 
            if(n == 0):
                ax1.plot(a_all[n], r_true_all[n], color=colors[2*n+1], lw=1, label='astropy')
            ax1.set_ylabel('LOS distance\n(Mpc/h)', fontsize=14)
        
        # plot comv dist vs scale factor error
        ax2.plot(a_all[n], r_all[n]-r_true_all[n], color=colors[2*n], lw=1.6, 
                 label=lcNames[n])
        if(n == 0):
            ax2.plot(a_all[n], r_true_all[n]-r_true_all[n], color=colors[2*n+1], lw=1, label='astropy')
        ax2.set_xlabel('a(t)', fontsize=16)
        ax2.set_ylabel('deviation from \nastropy result (Mpc/h)', fontsize=14)
        ax2.set_ylim([-3, 3])

        if(twoPanel):
            ax1.legend(loc="upper right")
            ax1.grid(True)
            ax2.grid(True)
        else:
            ax2.legend(loc="upper right")
            ax2.grid(True)
    
    if(plotMode == 'show'):
        plt.show()
    else:
        f.savefig('{}/lc_comvDist_v_redshift_{}-{}'.format(outDir, steps[0], steps[-1]))


#############################################################################################
#############################################################################################


def N_M_z(lcDir, minStep, z_bins = 4, mass_bins = 100, minMass = 1e13, 
          plotMode='show', outDir='.'):
    '''
    This function computes and plots the mass distribution of one lightcone output over a 
    chosen redshift range, N(M|z). The lightcone should have a valid "mass" column (a halo lightcone).
    The resulting plot is two vertical panels; one with the total distribution from z=0 to the 
    redshift corresponding to minStep, and one with overlapping histograms per redshift bin (given by
    z_bins over that range)

    Params:
    :param lcDir: A directory containing lightcone output. 
                   It is assumed that these directories follow the 
                   naming and structure convention given in fig 6 of 
                   the Creating Lightcones in HACC notes (with one 
                   subdirectory per snapshot). 
    :param maxStep: the minimum lightcone step to include
    :param z_bins: number of linear bins to use in redshift
    :param mass_bins: number of log bins to use in mass
    :param minMass: the minimum mass to include in the plot
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless otherwise
                     specified in the 'outDir' arg
    :param outDir: where to save figures, if plotMode == 'save'
    :return: None
    '''
    if plotMode not in ['show', 'save']:
        raise Exception('Unknown plotMode {}. Options are \'show\' or \'save\'.'.format(plotMode))
    
    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio

    # get lc subdirectory prefix of the input lightcone
    # (prefix could be 'lc', 'lcGals', or 'STEP', etc.). 
    print('working on lc at {}'.format(lcDir))
    subdirs = glob.glob('{}/*'.format(lcDir))
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue    

    # get available steps
    stepsAvail = np.sort([int(s.split('/')[-1].split(prefix)[-1]) for s in subdirs])
    stepsAvail = stepsAvail[stepsAvail >= minStep]

    step_bins = np.array_split(stepsAvail, z_bins)
    a = np.linspace(1/201.0, 1.0, 500)
    z =1/a-1
     
    # set up plotting
    config(cmap=plt.cm.plasma, numColors=z_bins)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    f = plt.figure(plt.gcf().number+1)
    
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)
    mass_bins = np.logspace(np.log10(minMass), 15.698970004336019, mass_bins)

    # define arrays to be plotted after looping through all step data.
    m_all = []
    z_all = []

    # start looping through arrays
    for j in range(z_bins):

        zbin = step_bins[j]
        m_bin_all = []
        z_bin_all = []
        
        for step in zbin:
            print('working on step {}'.format(step))
            
            # sort all files in this directory to get the gio header file
            lc = sorted(glob.glob('{0}/{1}{2}/*'.format(lcDir, prefix, step)))[0]
            
            # read and discard all objects below minMass
            m = np.squeeze(gio.gio_read(lc, 'mass'))
            m = m / wmap7.h
            massSort = np.argsort(m)
            m = m[massSort]
            cutoff_idx = np.searchsorted(m, minMass)
            m = m[cutoff_idx:]
            m_bin_all = np.hstack([m_bin_all, m])
            m_all = np.hstack([m_all, m])
             
        # plot N(M|z) for this bin
        ax2.hist( m_bin_all, bins=mass_bins, color=colors[j], lw=1, alpha=0.4, 
                label=r'${:.2f} \>\leq\> z \>\leq\> {:.2f}$'.format(z[zbin[-1]], z[zbin[0]]), zorder=10)
        ax2.hist( m_bin_all, bins=mass_bins, color=colors[j], lw=2, histtype = 'step', zorder=1) 
        ax2.set_yscale('log', nonposy='clip')
        ax2.set_xscale('log', nonposy='clip')
    
    # plot N(M|z) for all
    print('found {} total halos above mass {} Msun'.format(len(m_all), minMass))
    print('found {} total halos above mass 1e14 Msun'.format(len(m_all[m_all > 1e14])))
    print('found {} total halos above mass 1e15 Msun'.format(len(m_all[m_all > 1e15])))

    ax1.hist( m_all, bins=mass_bins, color=colors[0], lw=1, alpha=0.6, 
            label=r'${:.2f} \>\leq\> z \>\leq\> {:.2f}$'.format(0, z[min(stepsAvail)]))
    ax1.set_yscale('log', nonposy='clip')
    ax1.set_xscale('log', nonposy='clip')
    ax1.set_xlabel(r'$M_\mathrm{FOF}$', fontsize=14)
    ax1.set_ylabel(r'$N( \> M_\mathrm{{FOF}}  \>\> | \>\> z \>)$', fontsize=14)
    
    ax1.set_xlim([minMass, 5e15])
    ax1.legend(loc="upper right")
    ax1.grid(True)
        
    ax2.set_xlabel(r'$M_\mathrm{FOF}$', fontsize=14)
    ax2.set_ylabel(r'$N( \> M_\mathrm{{FOF}}  \>\> | \>\> z \>)$', fontsize=14)
    
    ax2.set_xlim([minMass, 5e15])
    ax2.legend(loc="upper right")
    ax2.grid(True)
    
    if(plotMode == 'show'):
        plt.show()
    else:
        plt.show()
        f.savefig('{}/N(M|z)_{}-{}.png'.format(outDir, z[max(stepsAvail)], z[min(stepsAvail)]))


#############################################################################################
#############################################################################################


def N_z(lcDirs, steps, lcNames=['interpolated + solver improvements', 'uncorrected'], 
        plotMode='show', outDir='.'):
    '''
    This function computes and plots the redshift distribution of at least one set of 
    lightcone output.

    Params:
    :param steps: A list of steps to include in the generated plot
    :param lcDirs: A list of directories containing lightcone output. 
                   It is assumed that these directories follow the 
                   naming and structure convention given in fig 6 of 
                   the Creating Lightcones in HACC notes (with one 
                   subdirectory per snapshot). 
    :param lcNames: A list of strings to represent the lightcone outputs
                   given in lcDirs. This is for plotting comparisons, 
                   to fill in the legend labels. The length of this array 
                   should of course match that of lcDirs
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless otherwise
                     specified in the 'outDir' arg
    :param outDir: where to save figures, if plotMode == 'save'
    :return: None
    '''
    if plotMode not in ['show', 'save']:
        raise Exception('Unknown plotMode {}. Options are \'show\' or \'save\'.'.format(plotMode))
    
    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio

    # get lc subdirectory prefix of each input lightcone
    # (prefix could be 'lc' or 'lcGals', etc.). 
    prefix = ['']*len(lcDirs)
    for j in range(len(prefix)):
        subdirs = glob.glob('{}/*'.format(lcDirs[j]))
        for i in range(len(subdirs[0].split('/')[-1])):
            try:
                (int(subdirs[0].split('/')[-1][i]))
                prefix[j] = subdirs[0].split('/')[-1][0:i]
                break
            except ValueError:
                continue
    
    # get step redshifts
    a = np.linspace(1./(200.+1.), 1., 500)
    z = 1./a-1.
    zs = z[steps]
    bins = np.linspace(min(zs), max(zs), 100)
    
    # set up plotting
    config(cmap=plt.cm.plasma, numColors=4)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    f = plt.figure(plt.gcf().number+1)
    
    ax2 = f.add_subplot(111)
    ax2.set_title('z = 0 -> {:.2f}'.format(max(zs)))

    # define arrays to be plotted after looping through all step data.
    # these will end up being arrays of array objects (NOT multidimensional
    # arrays, as the different lc outputs given by lcDirs probably have
    # different particle/object counts)
    a_all = [np.array([])]*len(lcDirs)

    # start looping through arrays
    for n in range(len(lcDirs)):
        lcDir = lcDirs[n]
        print('working on lc at {}'.format(lcDir))
        
        for step in steps:
            print('working on step {}'.format(step))
            
            # sort all files in this directory to get the gio header file
            lc = sorted(glob.glob('{0}/{1}{2}/*'.format(lcDir, prefix[n], step)))[0]
            
            # read and subsample
            a = gio.gio_read(lc, 'a')
            a_all[n] = np.hstack([a_all[n], np.squeeze(a)])
    
        # plot N(z)
        ax2.hist( (1/a_all[n]) - 1, bins=bins, color=colors[n*2], lw=1.3, alpha=0.8, 
                 label=lcNames[n], histtype='step')
        ax2.set_yscale('log', nonposy='clip')
        ax2.set_xlabel('z', fontsize=16)
        ax2.set_ylabel('N(z)', fontsize=16)

    ax2.legend(loc="upper left")
    ax2.grid(True)
    
    if(plotMode == 'show'):
        plt.show()
    else:
        f.savefig('{}/N(z)_{}-{}'.format(outDir, steps[0], steps[-1]))


#############################################################################################
#############################################################################################


def find_pairwise_separation(lcDir, lcName, saveDest, steps, fofDir, fofSubdirs=True, 
                             solverMode='forward', cosmology = 'alphaQ'):
    '''
    This function computes the pairwise distance - pairwise mass difference 
    phase space of all halo pairs in a lightcone output set. This is intended 
    to be run to check for duplicates in halo lightcones, as erroneous halo duplicates 
    should appear as an anomalous population in this distribution, with very small separation
    in spatial and mass space.

    Params:
    :param lcDir: A directory containing lightcone output. 
                  It is assumed that this directory follows the 
                  naming and structure convention given in fig 6 of 
                  the Creating Lightcones in HACC notes (with one 
                  subdirectory per snapshot).
    :param lcName: the suffix to use to identify the output file names
    :param saveDest: where to save the output numpy files
    :param steps: A list of steps to include in the halo pairing. Should include
                  at least two steps, since duplicate objects appear across lightcone
                  snapshot runs.
    :param fofDir: path to top-level directory of fof snapshot data from the simulation 
                   that was used to generate the data at lcDir.
    :param fofSubdirs: If true, assume that the snapshot data is grouped into
                       subdirectories of the format "STEPXXX". Otherwise, 
                       assume one flat directory with the the step number
                       "XXX" in the filenames somewhere, as "*.XXX.*" where
                       * is a wildcard
    :param solverMode: Wether the lightcone solver used to generate the output under 
                       validation was run in the standard forward mode (solverMode = 
                       'forward'), or the reversed mode, with extrapolation/interpolation
                       occuring backward in time (solverMode = 'backward').
    :param cosmology: simulation cosmology to use to compute NFW profiles (to get radii 
                      from masses)
    :return: None
    '''

    if solverMode not in ['forward', 'backward']:
        raise Exception('Unknown solver mode {}. Options are \'forward\' or \'backward\'.'.format(solverMode))
    
    steps = sorted(steps)

    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio
    from halotools.empirical_models import NFWProfile
    
    # define cosmology
    if(cosmology=='alphaQ'):
        cosmo = FlatLambdaCDM(H0=71, Om0=0.26479, Ob0=0.044792699861138666, Tcmb0 = 0, Neff = 0)
    else:
        raise Exception('Unknown cosmology, {}. Available is: alphaQ'.format(cosmology))

    # get lc subdirectory prefix of the input halo lightcone
    # (prefix could be 'lc' or 'lcGals', etc.). 
    prefix = ''
    subdirs = glob.glob('{}/*'.format(lcDir))
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue
     
    # start calculation
    print('working on lc at {}'.format(lcDir))
    
    x_all = np.array([])
    z_all = np.array([])
    y_all = np.array([])
    ids_all = np.array([])
    mass_all = np.array([])
    r200_all = np.array([])
    
    for j in range(len(steps)):

        step = steps[j]
        print('working on step {}'.format(step))
        if(step == 499): continue
        
        # sort all files in this directory to get the gio header file
        lc = sorted(glob.glob('{0}/{1}{2}/*'.format(lcDir, prefix, step)))[0]
        
        # read and subsample
        print('reading and subsampling')
        x = gio.gio_read(lc, 'x')
        y = gio.gio_read(lc, 'y')
        z = gio.gio_read(lc, 'z')
        a = gio.gio_read(lc, 'a')
        ids = gio.gio_read(lc, 'id')
     
        # Plot only duplicated halos
        dupl_file = h5.File('/home/hollowed/lc_halos_validation/duplData/duplicates_{}_{}.hdf5'.format(lcName, 
                                                                                                       steps[0])) 
        subsamp = np.in1d(ids, dupl_file['id_step{}'.format(step)][:])
        ordering = search_sorted(np.squeeze(dupl_file['id_step{}'.format(step)][:]), np.squeeze(ids[subsamp]))
        mmask = np.squeeze(gio.gio_read(lc, 'replication')[subsamp]) != np.squeeze(dupl_file['replication_step{}'.format(step)][:][ordering])

        #subsamp = np.random.choice(np.arange(len(x), dtype=int),
        #                           int(len(x)*0.001), replace=False)

        #x_all = np.hstack([x_all, np.squeeze(x[subsamp])])
        #y_all = np.hstack([y_all, np.squeeze(y[subsamp])])
        #z_all = np.hstack([z_all, np.squeeze(z[subsamp])])
        x_all = np.hstack([x_all, np.squeeze(x[subsamp])[mmask]])
        y_all = np.hstack([y_all, np.squeeze(y[subsamp])[mmask]])
        z_all = np.hstack([z_all, np.squeeze(z[subsamp])[mmask]])
        ids_all = np.hstack([ids_all, np.squeeze(ids[subsamp])[mmask]])

        # read tags from lc and fof catalog
        print('matching to fof catalog')
        if(fofSubdirs):
            if(solverMode == 'backward'):
                fof = sorted(glob.glob('{}/STEP{}/*fofproperties'.format(fofDir, steps[j+1])))[0]
            else:
                fof = sorted(glob.glob('{}/STEP{}/*fofproperties'.format(fofDir, step)))[0]
        else:
            if(solverMode == 'backward'):
                fof = sorted(glob.glob('{}/*.{}*fofproperties'.format(fofDir, steps[j+1])))[0]
            else:
                fof = sorted(glob.glob('{}/*.{}*fofproperties'.format(fofDir, step)))[0]
        
        fof_tags = np.squeeze(gio.gio_read(fof, 'fof_halo_tag'))
        #lc_tags = np.squeeze(gio.gio_read(lc, 'id'))[subsamp]
        lc_tags = np.squeeze(gio.gio_read(lc, 'id')[subsamp])[mmask]
        
        # mask lightcone fof tags to get rid of fragment signs and bits (has no effect 
        # for lightcones that already output standard fof tags)
        lc_tags = (np.sign(lc_tags) * lc_tags) & 0xffffffffffff
        
        # match to fof catalog and get masses
        lc_to_fof = search_sorted(fof_tags, lc_tags)
        if(np.sum(lc_to_fof == -1) != 0): 
            raise Exception('output in lightcone file {} not found in fof catalog, '\
                            'maybe passed the wrong files?'.format(lcDir))
        
        fof_mass = np.squeeze(gio.gio_read(fof, 'fof_halo_mass'))[lc_to_fof]
        mass_all = np.hstack([mass_all, fof_mass])

        # make nfw profile obj and find radii
        print('computing halo radii\n')
        nfw = NFWProfile(cosmology = cosmo, mdef = '200c')
        r200_all = np.hstack([r200_all, np.squeeze(nfw.halo_mass_to_halo_radius(fof_mass))])

    print('Done gathering data from all steps. \nFinding separations in space and mass, and'\
           ' maximum pair masses and radii\n\n')
    # find pariwise separations
    x_all = x_all[np.newaxis]
    y_all = y_all[np.newaxis]
    z_all = z_all[np.newaxis]
    mass_all = mass_all[np.newaxis]
    r200_all = r200_all[np.newaxis]
    xdiff = abs(x_all.T - x_all)
    ydiff = abs(y_all.T - y_all)
    zdiff = abs(z_all.T - z_all)
    
    # find pairwise mass difference
    massDiff = abs(mass_all.T - mass_all)

    # find mean radius and mass of each pair
    r200Mean = (r200_all.T + r200_all) / 2
    massMean = (mass_all.T + mass_all) / 2

    # take only upper traingle of above matrices to avoid double counting pairs
    # set the triangle offset to 1 ('k' arg) to remove the diagonal as well, which
    # will be pairs of identical halos
    triangleMask = np.triu_indices(np.shape(xdiff)[0], k=1)
    xdiff = xdiff[triangleMask]
    ydiff = ydiff[triangleMask]
    zdiff = zdiff[triangleMask]
    massDiff_all = massDiff[triangleMask]
    r200Mean_all = r200Mean[triangleMask]
    massMean_all = massMean[triangleMask]

    # get separation magnitude
    posDiff_all = np.sqrt(xdiff**2 + ydiff**2 + zdiff**2)
   
    pdb.set_trace()

    # done! save results to files
    print('Done. Saving to files at {}'.format(saveDest))
    np.save('{}/halo_pairMaxRadii_{}.npy'.format(saveDest, lcName), r200Mean_all)
    np.save('{}/halo_sep_{}.npy'.format(saveDest, lcName), posDiff_all)
    np.save('{}/halo_pairMaxMass_{}.npy'.format(saveDest, lcName), massMean_all)
    np.save('{}/halo_mass_diff_{}.npy'.format(saveDest, lcName), massDiff_all)
    
        
#############################################################################################
#############################################################################################


def plot_pairwise_separation(loadDest, lcName, steps, plotMode='save', outDir='.'):
    '''
    This function plots the pairwise distance - pairwise mass difference 
    phase space of all halo pairs in at least one lightcone output set. This is intended 
    to be run to check for duplicates in halo lightcones, as erroneous halo duplicates 
    should appear as an anomalous population in this distribution, with very small separation
    in spatial and mass space.

    Params:
    :param loadDest: where to load the numpy files output from find_pairwise_separation()
    :param lcName: the file name suffix that was used to save the data to be plotted, from a 
                   corresponding run of find_pairwise_separation()
    :param steps: A list of steps that were used in creating the data output by a corresponding 
                  run of find_pairwise_separation() 
    :param lcNames: A list of strings to represent the lightcone outputs given in lcDirs in 
                    a corresponding run of find_pairwise_separation. 
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless otherwise
                     specified in the 'outDir' arg
    :param outDir: where to save figures, if plotMode == 'save'
    :return: None
    '''
    
    # load data
    print('loading output of find_pairwise_separation at {}'.format(loadDest))
    posDiff_all = np.load('{}/halo_sep_{}.npy'.format(loadDest, lcName))
    r200Mean_all = np.load('{}/halo_pairMaxRadii_{}.npy'.format(loadDest, lcName))
    massDiff_all = np.load('{}/halo_mass_diff_{}.npy'.format(loadDest, lcName))
    massMean_all = np.load('{}/halo_pairMaxMass_{}.npy'.format(loadDest, lcName))

    # calculate quantities to bin (we want deltaM/M vs log10(1 + d/r200) )
    xData = massDiff_all / massMean_all
    yData = np.log10(1 + (posDiff_all) / r200Mean_all )
    
    # get step redshifts
    a = np.linspace(1./(200.+1.), 1., 500)
    z = 1./a-1.
    zs = z[steps]
    
    # set up plotting
    f = plt.figure(plt.gcf().number+1)
    
    ax = f.add_subplot(111)
    ax.set_title('z = 0 -> {:.2f} ({})'.format(max(zs), lcName))
    ax.set_xlabel('dM/M', fontsize=14)
    ax.set_ylabel('log10(1 + d/r200)', fontsize=14)
 
    # plot the pairwise phase space distribution
    hist = ax.hist2d(xData, yData, bins=200, cmap = plt.cm.plasma, norm=LogNorm())

    ax.grid()
    plt.colorbar(hist[3], ax=ax)

    if(plotMode == 'show'):
        plt.show()
    else:
        f.savefig('{}/pairwiseSep_{}-{}'.format(outDir, steps[0], steps[-1]))


#############################################################################################
#############################################################################################


def inspect_velocity_smoothing(lcDir, snapshotDir, steps, saveDest, lcName='out', sfrac=1.0):
    '''
    This function computes and saves the velocity difference of each object in a 
    lightcone output catalog, with repect to it's matching object in the simulation 
    snapshot data. This test is intended to check for the severity of velocity
    smoothing in the lightcone code (since we what is output from the interpolated
    lightcone routine is approximated linear velocities). The result is written out
    to numpy files.

    Params:
    :param lcDir: top-level directory of a lightcone output catalog, containing per
                  step subdirectories, as given in Fig.7 of the Creating Lightcone in 
                  HACC document
    :param snapshotDir: top-level directory of simulation snapshot outputs, assumed to
                        contain per-step subdirectories in the form of 'STEPXXX'
    :param steps: A list of output steps to include in the analysis
    :param sfrac: the lightcone object subsampling fraction
    :param lcName: the suffix to use to identify the output file names
    :param saveDest: where to save the output numpy files
    :return: None
    '''

    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio
    from halotools.empirical_models import NFWProfile
    
    steps = sorted(steps)

    # get lc subdirectory prefix of the input halo lightcone
    # (prefix could be 'lc' or 'lcGals', etc.). 
    prefix = ''
    subdirs = glob.glob('{}/*'.format(lcDir))
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue
     
    # start calculation
    print('working on lc at {}'.format(lcDir))
    
    vx_all = np.array([])
    vz_all = np.array([])
    vy_all = np.array([])
    ids_all = np.array([])
    
    for j in range(len(steps)):

        step = steps[j]
        print('working on step {}'.format(step))
        if(step == 499): continue
        
        # sort all files in this directory to get the gio header file
        lc = sorted(glob.glob('{0}/{1}{2}/*'.format(lcDir, prefix, step)))[0]
        
        # read lc and subsample
        print('reading and subsampling')
        vx = np.squeeze(gio.gio_read(lc, 'vx'))
        vy = np.squeeze(gio.gio_read(lc, 'vy'))
        vz = np.squeeze(gio.gio_read(lc, 'vz'))
        ids = np.squeeze(gio.gio_read(lc, 'id'))
     
        subsamp = np.random.choice(np.arange(len(vx), dtype=int),
                                   int(len(vx)*sfrac), replace=False)

        vx_all = np.hstack([vx_all, vx[subsamp]])
        vy_all = np.hstack([vy_all, vy[subsamp]])
        vz_all = np.hstack([vz_all, vz[subsamp]])
        ids_all = np.hstack([ids_all, ids[subsamp]])

        # read from snapshot catalog
        print('matching to snapshots')
        snapshot = sorted(glob.glob('{}/STEP{}/*'.format(snapshotDir, step)))[0]
        
        snapshot_ids = np.squeeze(gio.gio_read(snapshot, 'id'))
        
        # match lc to snapshots and get corresponding velocities
        lc_to_snapshot = search_sorted(snapshot_ids, ids[subsamp])
        
        if(np.sum(lc_to_snapshot == -1) != 0): 
            raise Exception('output in lightcone file {} not found in snapshot, '\
                            'maybe passed the wrong files?'.format(lcDir))
        
        snapshot_vx = np.squeeze(gio.gio_read(snapshot, 'vx'))[lc_to_snapshot]
        snapshot_vy = np.squeeze(gio.gio_read(snapshot, 'vy'))[lc_to_snapshot]
        snapshot_vz = np.squeeze(gio.gio_read(snapshot, 'vz'))[lc_to_snapshot]

    print('Done gathering data from all steps')
    print('Finding separations in velocity magnitude and direction\n')
    
    # find velocity mag and direction separations
    v_diff = (np.sqrt(vx**2 + vy**2 + vz**2) -  
              np.sqrt(snapshot_vx**2 + snapshot_vy**2 + snapshot_vz**2))
    v_ang = np.array([angle_between([vx[i], vy[i], vz[i]], 
                                     [snapshot_vx[i], snapshot_vy[i], snapshot_vz[i]]) 
                       for i in range(len(vx))])
                       
    # done! save results to files
    print('Done. Saving to files at {}'.format(saveDest))
    np.save('{}/lc_v_mag_diff_{}.npy'.format(saveDest, lcName), v_diff)
    np.save('{}/lc_v_ang_diff_{}.npy'.format(saveDest, lcName), v_ang)


#############################################################################################
#############################################################################################


def plot_velocity_smoothing(loadDest, steps, plotMode='show', outDir='.', absVals=False, 
                            lcName='out'):
    '''
    This function plots the output of inspect_velocity_smoothing() as a 2d histogram, 
    with one dimension being the velocity megnitude separation of all lightcone objects,
    and the other being the velocity direction angualr separation. 

    :param loadDest: Directory containing data output from inspect_velocity_smoothing
    :param steps: a list of lightcone steps that were included in a corresponding run of 
                  inspect_velocity_smoothing()
    :param plotMode: The plotting mode. Options are 'show' or 'save'. If 'show', the
                     figures will be opened upon function completion. If 'save', they
                     will be saved as .png files to the current directory, unless otherwise
                     specified in the 'outDir' arg
    :param outDir: where to save figures, if plotMode == 'save'
    :param absVals: whether or not to plot the absolute value of the velocity differences
    :param lcName: The identifying name at the end of the file naems output by
                   a corresponding run of inspect_velocity_smoothing()
    '''
    # load data
    print('loading output of inspect_velocity_smoothing() at {}'.format(loadDest))
    vMag_diff = np.load('{}/lc_v_mag_diff_{}.npy'.format(loadDest, lcName))
    vAng_diff = np.load('{}/lc_v_ang_diff_{}.npy'.format(loadDest, lcName))

    # take absolute value fo quantites
    if(absVals):
        vMag_diff = np.abs(vMag_diff)
        vAng_diff = np.abs(vAng_diff)
    
    # get step redshifts
    a = np.linspace(1./(200.+1.), 1., 500)
    z = 1./a-1.
    zs = z[steps]
    
    # set up plotting
    f = plt.figure(plt.gcf().number+1)
    
    ax = f.add_subplot(111)
    ax.set_title('steps {} -> {}\nz = 0 -> {:.2f} ({})'.format(max(steps), min(steps), 
                                                               max(zs), lcName))
    ax.set_xlabel(r'$|v_{\mathrm{lc}}| - |v_{\mathrm{snapshot}}|$', fontsize=14)
    ax.set_ylabel(r'$\mathrm{cos}^{-1}(\hat{v}_{\mathrm{lc}}\cdot\hat{v}_{\mathrm{snapshot}})$',
                  fontsize=14)
 
    # plot the phase space distribution
    hist = ax.hist2d(vMag_diff, vAng_diff, bins=100, cmap = plt.cm.plasma, norm=LogNorm())

    ax.grid()
    plt.colorbar(hist[3], ax=ax)

    if(plotMode == 'show'):
        plt.show()
    else:
        f.savefig('{}/velocitySep_{}-{}'.format(outDir, max(steps), min(steps)))


#############################################################################################
#############################################################################################


def plot_shell_density(lcDir, steps, saveDest, rL = 3000, reps = 2, sfrac=0.2):
    
    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where gio may not be available
    import genericio as gio
    
    steps = sorted(steps)[::-1]

    # get lc subdirectory prefix of the input halo lightcone
    # (prefix could be 'lc' or 'lcGals', etc.). 
    prefix = ''
    subdirs = glob.glob('{}/*'.format(lcDir))
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue
     
    # start calculation
    print('working on lc at {}'.format(lcDir))
    
    a = np.linspace(1/201.0, 1.0, 500)
    z = 1/a-1
    zs = z[steps]
    lc_extent = rL*reps

    for step in steps:
        
        if(step == 499): continue
       
        print('reading step {}'.format(step))
        lc_file = sorted(glob.glob('{}/{}{}/*'.format(lcDir, prefix, step)))[0]
        x = np.squeeze(gio.gio_read(lc_file, 'x'))
        y = np.squeeze(gio.gio_read(lc_file, 'y'))
        z = np.squeeze(gio.gio_read(lc_file, 'z'))
        
        downsMask = np.random.choice(np.arange(len(x)), int(len(x)*sfrac))

        binsx = np.linspace(0, lc_extent, 60)
        binsy = np.linspace(0, lc_extent, 60)
        
        plt.hist2d(x[downsMask], y[downsMask], bins = [binsx, binsy], cmap='plasma')
        plt.plot([0, lc_extent], [0, 0], '-k', lw=1)
        plt.plot([0, lc_extent], [lc_extent, lc_extent], '-k', lw=1)
        plt.plot([0, 0], [0, lc_extent], '-k', lw=1)
        plt.plot([lc_extent, lc_extent], [0, lc_extent], '-k', lw=1)
        plt.plot([0, lc_extent], [lc_extent/reps, lc_extent/reps], '-k', lw=1)
        plt.plot([lc_extent/reps, lc_extent/reps], [0, lc_extent], '-k', lw=1)
        plt.xlim([-lc_extent*0.05, lc_extent + lc_extent*0.05])
        plt.ylim([-lc_extent*0.05, lc_extent + lc_extent*0.05])
        plt.savefig('{}.png'.format(step))
        print('saved fig')
