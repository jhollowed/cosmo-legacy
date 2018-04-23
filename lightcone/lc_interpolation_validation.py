
import pdb
import glob
import h5py as h5
import numpy as np
from astropy.cosmology import FlatLambdaCDM

import matplotlib as mpl
from cycler import cycler
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

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
    colors = cmap(np.linspace(0.1, 0.9, numColors))
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


#############################################################################################
#############################################################################################

#######################################
#
#           Validation Tests
#
#######################################

def lightconeHistograms(lcDir1, lcDir2, step, rL, mode='particles', 
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

    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where dtk and/or gio may not be available
    import genericio as gio
    from dtk import sort
    
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
    else:
        raise Exception('mode parameter must be \'particles\' or \'halos\'')
        
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

    # do binary search using dtk utility to find each object from the interpolation
    # lc data in the extrapolation lc data
    print('matching arrays')
    matchMap = sort.search_sorted(eid[intersection_etoi], 
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
 
    # plot position, velocity, and redshift differences between interpolated
    # and extrapolated output as historgrams
    
    config(cmap=plt.cm.plasma)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bins = 300
    
    f = plt.figure(0) 
    ax =  f.add_subplot(311)
    ax.hist(posDiff, bins, color=colors[0])
    ax.set_yscale('log')
    ax.set_xlabel(r'$\left|\vec{r}_\mathrm{extrap} - \vec{r}_\mathrm{interp}\right|\>\>\mathrm{(Mpc/h)}$', fontsize=18)
    
    ax2 =  f.add_subplot(312)
    ax2.hist(mag_vDiff, bins, color=colors[1])
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\left|\vec{v}_\mathrm{extrap} - \vec{v}_\mathrm{interp}\right| \>\>\mathrm{(km/s)}$', fontsize=18)
    
    ax3 =  f.add_subplot(313)
    ax3.hist(redshiftDiff, bins, color=colors[2])
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$\left|z_\mathrm{extrap} - z_\mathrm{interp}\right|$', fontsize=18)
    
    if(plotMode == 'show'):
        plt.show()
    else:
        plt.savefig('{}/lc_diffHists_{}'.format(outDir, step))


#############################################################################################
#############################################################################################


def saveLightconePathData(epath, ipath, spath, outpath, rL, diffRange='max', 
                          mode='particles', snapshotSubdirs = False):
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
    
    For consistency, the time window used to inspect the objects paths is
    always steps 421 - 464, with step 442 being the one to read lightcone data
    from. It is assumed that the lightcone output directories follow the 
    naming and structure convention given in fig 6 of the Creating Lightcones
    in HACC notes. The strcture of the snapshot data directory can be given by
    the user in the snapshotSubdirectories parameter.
    
    This code is meant to be run on Datastar, or Cooley (somewhere with dtk compiled).
    
    Params:
    :param epath: path to a lightcone output directory generated by the 
                  extrapolation driver
    :param ipath: path to a lightcone output directory generated by the
                  interpolation driver. Should have been run with identical 
                  parameters, on the same cimulation volume, as the run that 
                  generated the data at epath
    :param spath: path to snapshot data from the simulation that was used to 
                  generate the data at epath and ipath
    :param outpath: where to write out the path data (npy files)
    :param rL: the box width of the simulation from which the lightcones at epath
               and ipath were generated
    :param diffRange: whether to use the 'max', 'med'(median) or 'min' diffVals
               and ipath were generated, in comoving Mpc/h
    :param mode: whether to perform the snapshot object match-up on particles or
                 halos. If mode=="particles", then find the object in each snapshot
                 by matching on it's 'id'. If mode=="halos", then find the object
                 in each snapshot by matching it's 'tree_node_index' and 
                 'desc_node_index'.
    :param snapshotSubdirs: If true, assume that the snapshot data is grouped into
                            subdirectories of the format "STEPXXX". Otherwise, 
                            assume one flat directory with the the step number
                            "XXX" in the filenames somewhere, as "*.XXX.*" where
                            * is a wildcard
    :return: None
    '''

    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where dtk and/or gio may not be available
    import genericio as gio
    from dtk import sort
    
    subdirs = glob.glob('{}/*'.format(ipath))
    
    # get lc subdirectory prefix (could be 'lc' or 'lcGals', etc.). 
    # prefix of subdirs in epath and ipath assumed to be the same.
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue

    # get file names in 442 subdir for interpolated and extrapolated lc data
    # (sort them to grab only the unhashed file header)
    ifile = sorted(glob.glob('{}/{}442/*'.format(ipath, prefix)))[0]
    efile = sorted(glob.glob('{}/{}442/*'.format(epath, prefix)))[0]
   
    # set id types to read based on the mode
    if(mode == 'particles'):
        idName = 'id'
    elif(mode == 'halos'):
        idName = 'tree_node_index'
    else:
        raise Exception('mode parameter must be \'particles\' or \'halos\'')
        
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
    ix = ix[initVolMask_interp]
    iy = iy[initVolMask_interp]
    iz = iz[initVolMask_interp]
    ia = ia[initVolMask_interp]
    irot = irot[initVolMask_interp]

    initVolMask_extrap = np.logical_and.reduce((abs(ex) < rL, 
                                               abs(ey) < rL, 
                                               abs(ez) < rL))
    eid = eid[initVolMask_extrap]
    ex = ex[initVolMask_extrap]
    ey = ey[initVolMask_extrap]
    ez = ez[initVolMask_extrap]
    ea = ea[initVolMask_extrap]
    erot = erot[initVolMask_extrap]

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

    # do binary search using dtk utility to find each object from the interpolation
    # lc data in the extrapolation lc data
    print('matching arrays')
    matchMap = sort.search_sorted(eid[intersection_etoi], 
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
    
    print("Reading snapshot files")
    # sort snapshot files and grad the first in the list so that we are
    # reading the un-hashed header file

    if(snapshotSubdirs): 
        sfile0 = sorted(glob.glob('{}/STEP421/*'.format(spath)))[0]
    else:
        sfile0 = sorted(glob.glob('{}/*.421*'.format(spath)))[0]
    sid0 = gio.gio_read(sfile0, idName)
    sx0 = gio.gio_read(sfile0, sx)
    sy0 = gio.gio_read(sfile0, sy)
    sz0 = gio.gio_read(sfile0, sz)
    if(mode == 'halos'):
        sdid0 = gio.gio_read(sfile0, 'desc_node_index')  

    if(snapshotSubdirs): 
        sfile1 = sorted(glob.glob('{}/STEP432/*'.format(spath)))[0]
    else:
        sfile1 = sorted(glob.glob('{}/*.432*'.format(spath)))[0]
    sid1 = gio.gio_read(sfile1, idName)
    sx1 = gio.gio_read(sfile1, sx)
    sy1 = gio.gio_read(sfile1, sy)
    sz1 = gio.gio_read(sfile1, sz)
    if(mode == 'halos'):
        sdid1 = gio.gio_read(sfile1, 'desc_node_index')  
    
    if(snapshotSubdirs): 
        sfile2 = sorted(glob.glob('{}/STEP442/*'.format(spath)))[0]
    else:
        sfile2 = sorted(glob.glob('{}/*.442*'.format(spath)))[0]
    sid2 = gio.gio_read(sfile2, idName)
    sx2 = gio.gio_read(sfile2, sx)
    sy2 = gio.gio_read(sfile2, sy)
    sz2 = gio.gio_read(sfile2, sz)
    if(mode == 'halos'):
        sdid2 = gio.gio_read(sfile2, 'desc_node_index')  
    
    if(snapshotSubdirs): 
        sfile3 = sorted(glob.glob('{}/STEP453/*'.format(spath)))[0]
    else:
        sfile3 = sorted(glob.glob('{}/*.453*'.format(spath)))[0]
    sid3 = gio.gio_read(sfile3, idName)
    sx3 = gio.gio_read(sfile3, sx)
    sy3 = gio.gio_read(sfile3, sy)
    sz3 = gio.gio_read(sfile3, sz)
    if(mode == 'halos'):
        sdid3 = gio.gio_read(sfile3, 'desc_node_index')  
    
    if(snapshotSubdirs): 
        sfile4 = sorted(glob.glob('{}/STEP464/*'.format(spath)))[0]
    else:
        sfile4 = sorted(glob.glob('{}/*.464*'.format(spath)))[0]
    sid4 = gio.gio_read(sfile4, idName)
    sx4 = gio.gio_read(sfile4, sx)
    sy4 = gio.gio_read(sfile4, sy)
    sz4 = gio.gio_read(sfile4, sz)
    if(mode == 'halos'):
        sdid4 = gio.gio_read(sfile4, 'desc_node_index')  

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
        this_erot = erot[iMask][idx]

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
        if(mode == 'halos'):
            # match the halo tree node index to the node index 
            # and descendent node index of steps 442 and 432, 
            # respectively
            s1_idx = np.where(sdid1 == iid[iMask][idx])
            s2_idx = np.where(sid2 == iid[iMask][idx])
            # match the tree and descendent node idecies to those
            # found above
            s0_idx = np.where(sdid0 == sid1[s1_idx])
            s3_idx = np.where(sid3 == sdid2[s2_idx])
            s4_idx = np.where(sid4 == sdid3[s3_idx])

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
       
        # approximated particle paths for steps 442-453 
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
    config(cmap=plt.cm.cool)

    # get data files
    data = '{}/{}_diff'.format(dataPath, diffRange)
    numFiles = len(glob.glob('{}/truex_*'.format(data)))

    # loop through all files and plot (each file corresponds to one 
    # lightcone object)
    for i in range(numFiles):

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

def findDuplicates(lcDir, steps, lcSuffix, outDir):
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
    :param steps: an array of two lightcone outputs, by snapshot number, between 
                  which to check for duplicate particles/objects.
    :param lcSuffix: An identifier string that will be used as a filename suffix
                     for the output hdf5 files. This can be used to distinguish 
                     between multiple lightcones within which duplicates will be 
                     searched for.
    :param outDir: Location to save the output hdf5 file
    :return: None
    '''
    if(len(steps) != 2):
        raise Exception('Only two step numbers should be passed in the \'steps\' arg')

    # do these imports here, since there are other functions in this file that
    # are intended to run on systems where dtk and/or gio may not be available
    import genericio as gio
    from dtk import sort
    
    subdirs = glob.glob('{}/*'.format(lcDir))
    
    # get lc subdirectory prefix (could be 'lc' or 'lcGals', etc.). 
    # prefix of subdirs in epath and lcDir assumed to be the same.
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue

    # sort the subdirectory contents and take the first item, since that
    # should always be the GIO header file
    print('reading data')
    file1 = sorted(glob.glob('{}/{}{}/*'.format(lcDir, prefix, steps[0])))[0]
    file2 = sorted(glob.glob('{}/{}{}/*'.format(lcDir, prefix, steps[1])))[0]
    outfile = h5.File('{}/duplicates_{}.hdf5'.format(outDir, lcSuffix), 'w')

    ids1 = gio.gio_read(file1, 'id')
    ids1 = np.ndarray.flatten(ids1)
    ids2 = gio.gio_read(file2, 'id')
    ids2 = np.ndarray.flatten(ids2)
    
    print('matching')
    matches = sort.search_sorted(ids1, ids2)

    matchesMask2 = matches != -1
    matchesMask1 = matches[matchesMask2]

    print('found {} duplicates'.format(np.sum(matchesMask2)))

    dup_ids1 = ids1[matchesMask1]
    x1 = np.squeeze(gio.gio_read(file1, 'x')[matchesMask1])
    y1 = np.squeeze(gio.gio_read(file1, 'y')[matchesMask1])
    z1 = np.squeeze(gio.gio_read(file1, 'z')[matchesMask1])
    
    dup_ids2 = ids2[matchesMask2]
    x2 = np.squeeze(gio.gio_read(file2, 'x')[matchesMask2])
    y2 = np.squeeze(gio.gio_read(file2, 'y')[matchesMask2])
    z2 = np.squeeze(gio.gio_read(file2, 'z')[matchesMask2])
    
    repeat_frac = float(len(dup_ids2)) / len(ids2) 
    print('repeat fraction is {}'.format(repeat_frac))

    print('writing out {}'.format('{}/duplicates_{}.hdf5'.format(outDir, lcSuffix)))
    outfile.create_dataset('repeat_frac', data=np.array([repeat_frac]))
    outfile.create_dataset('id', data = np.hstack([dup_ids2, dup_ids1]))
    outfile.create_dataset('x', data = np.hstack([x2, x1]))
    outfile.create_dataset('y', data = np.hstack([y2, y1]))
    outfile.create_dataset('z', data = np.hstack([z2, z1]))


#############################################################################################
#############################################################################################


def compareDuplicates(duplicatePath, steps, lcSuffix, plotMode='show', outDir='.'):
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
    :param steps: an array of two lightcone outputs, by snapshot number, between 
                  which a corresponding run of finDuplicates() checked for duplicate 
                  particles/objects.
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
        raise Exception('Input array lcSuffix should be of legnth 2. See function docstrings')
    if(len(steps) != 2):
        raise Exception('Only two step numbers should be passed in the \'steps\' arg')

    # open files
    dupl1 = h5.File('{}/duplicates_{}.hdf5'.format(duplicatePath, lcSuffix[0]), 'r')
    dupl2 = h5.File('{}/duplicates_{}.hdf5'.format(duplicatePath, lcSuffix[1]), 'r')

    print('Duplicate fraction for {} output: {}'.format(lcSuffix[0], dupl1['repeat_frac'][:][0]))
    print('Duplicate fraction for {} output: {}'.format(lcSuffix[1], dupl2['repeat_frac'][:][0]))
    
    # setup plotting
    f = plt.figure(1)
    axe = f.add_subplot(121, projection='3d')
    axi = f.add_subplot(122, projection='3d')
    title = f.suptitle('step {} - step {}'.format(steps[0], steps[1]))
    axi.set_title('{}\nDuplicate fraction: {:.2E}'.format(lcSuffix[0], dupl1['repeat_frac'][:][0]))
    axe.set_title('{}\nDuplicate fraction: {:.2f}'.format(lcSuffix[1], dupl2['repeat_frac'][:][0]))

    # find intersection and symmetric difference of the two outputs
    maski = np.in1d(dupl1['id'], dupl2['id'])
    maske = np.in1d(dupl2['id'], dupl1['id'])
    maske_nokeep = np.random.choice(np.where(~maske)[0], 
                                    int(len(np.where(~maske)[0])*0.9), replace=False)
    maske[maske_nokeep] = 1
    e_downsample_idx = np.random.choice(np.linspace(0, len(dupl2['id'][:])-1, 
                                        len(dupl2['id'][:]), dtype=int), 
                                        int(len(dupl2['id'][:])*0.1), replace=False)
    e_downsample = np.zeros(len(dupl2['id'][:]), dtype = bool)
    e_downsample[e_downsample_idx] = 1
    
    # do plotting. The extrapolation is downsampled, while the interpolated output
    # is not, since the extrapolated output should have far mroe duplicate objects
    axe.plot(dupl2['x'][e_downsample], dupl2['y'][e_downsample], dupl2['z'][e_downsample], 
            '.g', ms=1, label='shared duplicates')
    axe.plot(dupl2['x'][~maske], dupl2['y'][~maske], dupl2['z'][~maske], 
             '+m', mew=1, label='unique duplicates')
    axe.set_xlabel('x (Mpc/h)')
    axe.set_ylabel('y (Mpc/h)')
    axe.set_zlabel('y (Mpc/h)')
    axe.legend()

    axi.plot(dupl1['x'], dupl1['y'], dupl1['z'], 
             '.b', ms=1, label='shared duplicates')
    axi.plot(dupl1['x'][~maski], dupl1['y'][~maski], dupl1['z'][~maski], 
             '+r', mew=1, label='unique duplicates')
    axi.set_xlabel('x (Mpc/h)')
    axi.set_ylabel('y (Mpc/h)')
    axi.set_zlabel('y (Mpc/h)')
    axi.legend()

    if(plotMode == 'show'):
        plt.show()
    else:
        f.savefig('{}/lc_duplicates_{}-{}'.format(outDir, steps[0], steps[1]))


#############################################################################################
#############################################################################################


def compareReps(lcDir1, lcDir2, step, plotMode='show', outDir='.'):
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
    # are intended to run on systems where dtk and/or gio may not be available
    import genericio as gio
   
    # setup plottong
    f = plt.figure(2)
    ax1 = f.add_subplot(221, projection='3d')
    ax2 = f.add_subplot(223, projection='3d')
    
    # find 
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

    rot64 = gio.gio_read(file1, 'rotation')
    rep64 = gio.gio_read(file1, 'replication')
    rot256 = gio.gio_read(file2, 'rotation')
    rep256 = gio.gio_read(file2, 'replication')

    uniqRep64 = sorted(np.unique(rep64))
    uniqRep256 = sorted(np.unique(rep256))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    for j in range(len(uniqRep64)):

        xReps64 = -((uniqRep64[j] >> 20) - 1) * 256
        yReps64 = -(((uniqRep64[j] >> 10) & 0x3ff) - 1) * 256
        zReps64 = -((uniqRep64[j] & 0x3ff) - 1) * 256
        rot1 = rot64[np.where(rep64 == uniqRep64[j])][0]
        if(np.sum(abs(np.diff(rot64[np.where(rep64 == uniqRep64[j])]))) != 0): 
            print('shit')
            return

        xReps256 = -((uniqRep256[j] >> 20) - 1) * 256
        yReps256 = -(((uniqRep256[j] >> 10) & 0x3ff) - 1) * 256
        zReps256 = -((uniqRep256[j] & 0x3ff) - 1) * 256
        rot2 = rot256[np.where(rep256 == uniqRep256[j])][0]
        if(np.sum(abs(np.diff(rot256[np.where(rep256 == uniqRep256[j])]))) != 0): 
            print('shit')
            return

        plotBox(xReps64, yReps64, zReps64, 256, 256, 256, ax1, colors[rot1])
        plotBox(xReps256, yReps256, zReps256, 256, 256, 256, ax2, colors[rot2])

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    if(plotMode == 'show'):
        plt.show()
    else:
        f.savefig('{}/lc_boxRotations_{}'.format(outDir, step))


#############################################################################################
#############################################################################################


def comvDist_vs_z(lcDirs, steps, lcNames=['second order corrections w/ weighting', 'uncorrected'], 
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
    # are intended to run on systems where dtk and/or gio may not be available
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
    f = plt.figure(3)
    
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
