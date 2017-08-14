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
import dtk
import coreTools as ct
import dispersionStats as stat
import time
from astropy.cosmology import WMAP7 as cosmo

def group_halos(catalog = 0, clusterMass = 1e14, disruption = 0.06, 
                massThresh = 10**11.6, process = True, timed = True, maskRad = True):
    '''
    Core catalogs produced from simulations are saved in a single large data table, with one
    column being a cores parent-halo id. This function groups cores into individual numpy files 
    according to their parent halo, groups those halos by step, and saves everything to a single 
    organized hdf5 file. In this way, the core data is considerably easier to manage at later 
    stages of analysis. 
    The cores are also processed through this function as well, if enabled, which means that any 
    falling outside of the radius and mass cuts are removed from the data. 
    Velocities and positions are *not* with respect to the host cluster after this data managing - 
    they are relative to the simulation box. Positions are also *not* unwrapped - meaning that 
    any halos near the simulation boundary may have halos that wrapped around to the other side of 
    the box, and therefore have vastly different positions. 

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
    zTool = dtk.StepZ(200, 0, 500)

    # locate catalog data, build save file
    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    p_prefix = ['un', ''][process]
    corePath = '/media/luna1/rangel/AlphaQ/CoreCat/{}'.format(catalogName)
    sodPath = '/media/luna1/dkorytov/data/AlphaQ/sod'    
    saveDest = 'data/coreCatalog/{}'.format(catalogName)
    if(maskRad):
        outputFile = h5py.File('{}/haloCores_{}processed_masked.hdf5'.format(saveDest, p_prefix), 'w')
    else:
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
                print('saving big halo with {}/{} cores ({})'
                       .format(sum(coreMask),len(haloMask),bigHalos))
            else:
                print('saving big halo with {} cores ({})'.format(len(haloMask), bigHalos))


            # save core data to halo file
            core_cols = ['core_tag', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radius', 
                         'infall_mass', 'infall_step', 'infall_fof_halo_tag']
            for i in range(len(core_cols)):
                nextData = gio.gio_read(cores, core_cols[i])[haloMask]
                if(process): nextData = nextData[coreMask]
                nextHalo.create_dataset(core_cols[i], data=nextData)

            # save halo properties to hdf5 group attributes
            halo_cols = ['fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z', 
                         'sod_halo_min_pot_x', 'sod_halo_min_pot_y', 'sod_halo_min_pot_z', 
                         'sod_halo_mean_vx', 'sod_halo_mean_vy', 'sod_halo_mean_vz', 
                         'sod_halo_radius', 'sod_halo_mass', 'sod_halo_vel_disp', 
                         'core_vel_disp', 'core_vel_disp_err', 'core_vel_disp_masked', 
                         'core_vel_disp_err_masked']
            for i in range(len(halo_cols)-4):
                nextData = gio.gio_read(halos, halo_cols[i])[k]
                nextHalo.attrs.create(halo_cols[i], data=nextData)
            
            # calculate core velocity dispersion and save as attribute
            halo_v = np.array([nextHalo.attrs['sod_halo_mean_v{}'.format(v)] 
                               for v in ['x','y','z']])
            core_vs = np.rec.fromarrays([nextHalo['vx'][:] - halo_v[0], 
                                         nextHalo['vy'][:] - halo_v[1],
                                         nextHalo['vz'][:] - halo_v[2]], 
                                         names = ['vx', 'vy', 'vz'])
            [core_velDisp, core_velDispErr] = ct.core_velDisp(core_vs, err=True)
            nextHalo.attrs.create(halo_cols[-4], core_velDisp)
            nextHalo.attrs.create(halo_cols[-3], core_velDispErr)
            
            if(maskRad):

                z = zTool.get_z(step) 
                a = cosmo.scale_factor(z) 

                # calculate core velocity dispersion excluding central cores within 0.1r200
                r_cores = np.array([nextHalo['x'][:], nextHalo['y'][:], nextHalo['z'][:]]).T
                r_halo = np.array([nextHalo.attrs['sod_halo_min_pot_{}'.format(p)] 
                                   for p in ['x','y','z']])
                r_cores = ct.unwrap_position(r_cores, r_halo)
                r_cores = r_cores * a
                r_halo = r_halo * a
                r_rel = r_cores - r_halo
                r_rel_mag = np.linalg.norm(r_rel, axis=1)
                r_norm = r_rel_mag / (nextHalo.attrs['sod_halo_radius'])

                rMask = (r_norm > 0.1)
                core_vs_masked = core_vs[rMask]
                [core_velDisp_masked, core_velDispErr_masked] = ct.core_velDisp(core_vs_masked,err=True)
                nextHalo.attrs.create(halo_cols[-2], core_velDisp_masked)
                nextHalo.attrs.create(halo_cols[-1], core_velDispErr_masked)
            
        if(bigHalos == 0):
            print('no cluster-sized halos')    
            continue

    if timed: 
        end = time.time()
        totalTime = end - start
        print('gio2hdf took {} s'.format(totalTime))
