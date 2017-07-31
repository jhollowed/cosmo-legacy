'''
Joe Hollowed
COSMO-HEP 2017
'''
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import pdb
import glob
import h5py
import numpy as np
from dtk import gio
import coreTools as ct
import dispersionStats as stat
import time

def group_halos(catalog = 0, clusterMass = 1e14, disruption = 0.06, 
                massThresh = 10**11.6, process = True, timed = False):
    '''
    Core catalogs produced from simulations are saved in a single large data table, with one
    column being a cores parent-halo id. This function groups cores into individual numpy files 
    according to their parent halo, groups those halos by step, and saves everything to a single 
    organized hdf5 file. In this way, the core data is considerably easier to manage at later 
    stages of analysis. The cores are also processed through this function as well, if enabled.

    :param catalog: which catalog to use; 0=BLEVelocity, 1=MedianVelocity, 2=CentralVelocity
    :param clusterMass: only save info from cluster-sized halos above this mass cut
    :param disruption: radius above which a core will be considered "disrupted" and discarded, 
               if process == True
    :param massThresh: infall mass below which a core will be considered too small to host a 
               galaxy and discarded, if process == True
    :param process: whether or not to process the cores
    :param timed: whether or not to time this function and print results
    :return: nothing, save hdf5 file
    '''
    if timed: start =  time.time()

    # locate catalog data, build save file
    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    p_prefix = ['un', ''][process]
    corePath = '/media/luna1/rangel/AlphaQ/CoreCat/{}'.format(catalogName)
    sodPath = '/media/luna1/dkorytov/data/AlphaQ/sod'    
    saveDest = '/home/jphollowed/data/hacc/alphaQ/coreCatalog/{}'.format(catalogName)
    outputFile = h5py.File('{}/haloCores_{}processed.hdf5'.format(saveDest, p_prefix), 'w')
    outputFile.attrs.create('disruption radius', disruption)
    outputFile.attrs.create('infall mass cut', massThresh)

    # gather all catalog files and sort by simulation step
    allCores = np.array(glob.glob('{}/*.coreproperties'.format(corePath)))
    steps = [int(f.split('.')[-2]) for f in allCores]
    sortOrder = np.argsort(steps)
    allCores = allCores[sortOrder]
    
    allHalos = np.array(glob.glob('{}/*.sodproperties'.format(sodPath)))
    steps = [int(f.split('.')[-2].split('-')[-1]) for f in allHalos]
    sortOrder = np.argsort(steps)
    allHalos = allHalos[sortOrder]
    
    
    # begin looping through each simulation step
    for j in range(len(allCores)):

        bigHalos = 0
        cores = allCores[j]
        halos = allHalos[j]
        step = np.array(steps)[sortOrder][j]
        print('\n---------- WORKING ON STEP {}({}) ----------'.format(step, j))
        
        # create sub-group for this snapshot in output hdf5
        nextSnapshot = outputFile.create_group('step_{}'.format(step))

        # find unique halos tags among core data. Proceed in saving core data for any
        # cluster sized halos (as defined by the parameter 'clusterMass')
        core_parentTags = gio.gio_read(cores, 'fof_halo_tag')
        core_parentTags = ct.mask_tags(core_parentTags)
        
        haloTags = gio.gio_read(halos, 'fof_halo_tag')
        haloMass = gio.gio_read(halos, 'sod_halo_mass')

        if(len(core_parentTags) == 0): 
            print('no halos with cores...')
            continue

        # begin looping through each halo in current step
        for k in range(len(haloTags)):

            # if this halo is below the mass cut, skip it, otherwise find all member cores
            if(haloMass[k] < clusterMass): continue
            bigHalos += 1
            haloMask = np.where(core_parentTags == haloTags[k])[0]
            
            # create sub-group for this snapshot in output hdf5
            nextHalo = nextSnapshot.create_group('halo_{}'.format(haloTags[k]))

            if(process):
                # discard disrupted & small mass cores
                core_radii = gio.gio_read(cores, 'radius')[haloMask]
                core_infallMass = gio.gio_read(cores, 'infall_mass')[haloMask]
                radMask = core_radii < disruption
                massMask = core_infallMass > massThresh
                coreMask = np.logical_and(radMask, massMask)
                #unwrap positions
                core_pos = np.array([gio.gio_read(cores, p)[haloMask] for p in ['x','y','z']]).T
                halo_pos = np.array([gio.gio_read(halos, 'sod_halo_min_pot_{}'.format(p))[k] 
                                     for p in ['x', 'y', 'z']]).T
                core_pos = ct.unwrap_position(core_pos, halo_pos).T
                print('saving big halo with {}/{} cores ({})'
                       .format(sum(coreMask),len(haloMask),bigHalos))
            else:
                print('saving big halo with {} cores ({})'.format(len(haloMask), bigHalos))


            # save core data to halo file
            core_cols = ['core_tag', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radius', 
                         'infall_mass', 'infall_step', 'infall_fof_halo_tag']
            for i in range(len(core_cols)):
                nextData = gio.gio_read(cores, core_cols[i])[haloMask]
                if(process): 
                    nextData = nextData[coreMask]
                    if(i in [1, 2, 3]): nextData = core_pos[i-1]
                nextHalo.create_dataset(core_cols[i], data=nextData)

            # save halo properties to hdf5 group attributes
            halo_cols = ['fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z', 
                         'sod_halo_min_pot_x', 'sod_halo_min_pot_y', 'sod_halo_min_pot_z', 
                         'sod_halo_mean_vx', 'sod_halo_mean_vy', 'sod_halo_mean_vz', 
                         'sod_halo_radius', 'sod_halo_mass', 'sod_halo_vel_disp', 
                         'core_vel_disp', 'core_vel_disp_err']
            for i in range(len(halo_cols)-2):
                nextData = gio.gio_read(halos, halo_cols[i])[k]
                nextHalo.attrs.create(halo_cols[i], data=nextData)
            
            # calculate core velocity dispersion and save as attribute
            halo_v = np.array([nextHalo.attrs['sod_halo_mean_v{}'.format(v)] 
                               for v in ['x','y','z']])
            core_vs = np.rec.fromarrays([nextHalo['vx'].value - halo_v[0], 
                                         nextHalo['vy'].value - halo_v[1],
                                         nextHalo['vz'].value - halo_v[2]], 
                                         names = ['vx', 'vy', 'vz'])
            [core_velDisp, core_velDispErr] = ct.core_velDisp(core_vs, err=True)
            nextHalo.attrs.create(halo_cols[-2], core_velDisp)
            nextHalo.attrs.create(halo_cols[-1], core_velDispErr)
            
        if(bigHalos == 0):
            print('no cluster-sized halos')    
            continue

    if timed: 
        end = time.time()
        totalTime = end - start
        print('gio2hdf took {} s'.format(totalTime))
