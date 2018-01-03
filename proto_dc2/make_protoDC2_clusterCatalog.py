'''
Joe Hollowed 
COSMO-HEP 2017

There are two functions here (each described in detail in their docstrings), makeCatalog() and
makeCatalog_steps(). The former operates on the protoDC2 catalog, which is produced via a
a light cone, and the latter operates on the upstream step-version of the protoDC2 catalog.
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import pdb
import glob
import time
import h5py
import numpy as np
from dtk import gio
import clstrTools as ct
import matplotlib as mpl
import dispersionStats as stat
import matplotlib.pyplot as plt


def makeCatalog():
    '''
    This function takes the protoDC2 galaxy catalog and reformats it as a cluster catalog, structured 
    as an hdf5 file as follows: 

    - The root group has two main groups, '/metaData' and '/clusters' 
    - The simulation (cosmology) parameters, galacticus parameters, and other catalog metadata are all
           given in the group '/metaData'
    - The '/clusters' group has a child group for each halo of mass > 1e14, the names of which being
      an incrementing integer. For example, the first halo found in this code with tag 123456 is found 
      at '/halo_1' in the hdf file, with the 'fof_halo_tag' group attribute set to '123456'
    - If duplicates of a halo were found (or at least, the same tag was used again in a later snapshot),
      then they are both saved and can be differentiated by the group attribute 'halo_step' 
    - Each halo group contains datasets which correspond to the data columns of the original 
      protoDC2 catalog's '/galaxyProperties' group, with consistent column names, for all of that halo's 
      member galaxies. If a halo is the host to 120 galaxies, then each of the data columns are 120 elements 
      long.
    - The '/galaxyProperties' group in the original catalog also has a few child groups, including '/morphology', 
      etc. These are still maintained, and present for each halo (the morphology porperties of the galaxies 
      belonging to some halo at index i would be found in '/clusters/halo_i/morphology')
    - A few datasets from the original catalog are missing from the halo group datasets. This is because
      the galaxy catalog has a few columns that apply halo-wide. For instance, the hostHaloMass is the 
      same value for every galaxy in the halo. Rather than repeating the information for every member
      galaxy, halo-wide properties are saved as group attributes. In addition to this data, the SO 
      properties of each halo are also saved as attributes, as well as a few other quantities 
      calculated in this code. Each halo group has the following attributes:
      
      From protoDC2:
        -   hostHaloMass (equivalent to the merger-tree mass (the FOF mass minus any fragments))
        -   hostHaloTag  (the FOF tag)
        -   hostIndex
        -   step
        -   halo_ra   (RA of the central galaxy)
        -   halo_dec  (Dec of the central galaxy)
      From the SO catalog:
        -   sod_halo_mass      (m200)
        -   sod_halo_radius    (r200)
        -   sod_halo_vel_disp  (km/s)
        -   sod_halo_ke
        -   sod_halo_cdelta
        -   sod_halo_cdelta_error
        -   sod_halo_c_acc_mass
      Calculated here:
        -   halo_z    (median of member galaxies observed redshifts) 
                       (combination of cosmological + peculiar))
        -   halo_z_err (standard error of the median (1.25 * SDOM) of the above quantity 'z' )
        -   gal_vel_disp_obs - the "observed" velocity dispersion of the halo's protoDC2 galaxies, 
                               obtained using only the derived observed redshift of each galaxy
        -   gal_vel_disp_1d  - the 1d velocity dispersion of the halo's protoDC2 galaxies, obtained
                               using their 3-dimensional velocity vector components (Evrard+ 2003)

        ** If a more robust statistic turns out to be needed for the first three calculated quanties 
           above, or the redshift error ought to be simulated rather than assumed from a normal 
           distribution, then I will make those changes. But nearly all halos in this catalog have 
           large numbers of member galaxies (on order 10^3) **

    Finally - this code as it stands should be run on datastar
    '''

# ====================================================================================================

    start = time.time()
    
    # load halo sod catalog
    haloCatalogPath = '/media/luna1/dkorytov/data/AlphaQ/sod'
    haloCatalog = glob.glob('{}/*.sodproperties'.format(haloCatalogPath))
    haloSteps = np.array([int(f.split('.')[0].split('-')[-1]) for f in haloCatalog])

    # define desired sod halo properties
    sodProps = ['sod_halo_radius', 'sod_halo_mass', 'sod_halo_ke', 'sod_halo_vel_disp', 
                'sod_halo_cdelta', 'sod_halo_cdelta_error', 'sod_halo_c_acc_mass']
    
    # load protoDC2 catalog
    protoDC2Path='/media/luna1/dkorytov/projects/protoDC2/output/versions/mock_v2.1.1.hdf5'
    #protoDC2Path='/media/luna1/dkorytov/projects/protoDC2/output/'\
    #             'mock_full_nocut_dust_elg_shear3_test.hdf5'
    path_out = '/media/luna1/jphollowed/protoDC2/versions/mock_clusters_v2.1.1.hdf5'
    #path_out = '/media/luna1/jphollowed/protoDC2/mock_clusters_v2.1.1_test_noattrs.hdf5'
    protoDC2 = h5py.File(protoDC2Path, 'r')
    outputFile = h5py.File(path_out, 'w')

    # -------- OLD --------
    # copy global attributes from original catalog
    #globalAttrs = list(protoDC2.attrs.keys())
    #for attr in globalAttrs:
    #    outputFile.attrs.create(attr, protoDC2.attrs[attr])

    # copy metaData group from original catalog
    output_clusters = outputFile.create_group('clusters')
    output_metaData = outputFile.create_group('metaData')
    metaData = protoDC2['metaData']
    metaData_keys = []
    metaData.visit(metaData_keys.append)
    metaData_keys = np.array(metaData_keys)
    metaData_objType = np.array([isinstance(metaData[key], h5py.Group) for key in metaData_keys])
    metaData_groups = metaData_keys[metaData_objType]
    metaData_datasets = metaData_keys[~metaData_objType]
    for group in metaData_groups: output_metaData.create_group(group)
    for dataset in metaData_datasets: output_metaData.create_dataset(dataset, data=metaData[dataset])

    # find all unique halo tags in protoDC2
    protoDC2g = protoDC2['galaxyProperties']
    massMask = protoDC2g['hostHaloMass'][:] > 1e14 
    uniqueTags = np.unique(protoDC2g['hostHaloTag'][:][massMask], return_counts = True)
    halos = uniqueTags[0]
    nGals = uniqueTags[1]
    duplicates=0
    nHalos = 1

    # ---------------------------------------------------------------------------------------

    # loop through all halos
    for k in range(len(halos)):
        print('\nworking on halo {} ({}/{})'.format(halos[k], k+1, len(halos)))

        # boolean mask of current halo galaxy membership
        galMask = [protoDC2g['hostHaloTag'][:] == halos[k]]

        # -----------------------------------------------------------------------------------
        # check for duplication

        hostSteps = protoDC2g['step'][:][galMask[0]] 
        if( sum(abs(np.diff(hostSteps))) != 0):
            duplicates += 1
            print('Found duplicate ({})'.format(duplicates))
            uniqueSteps = np.unique(hostSteps)
            galMask = np.array([np.logical_and(galMask[0], protoDC2g['step'][:] == step) 
                       for step in uniqueSteps])
            
            # make sure all duplicates still fit the cluster mass criteria
            duplMasses = [protoDC2g['hostHaloMass'][:][mask] for mask in galMask]
            
            for masses in duplMasses:
                if( sum(abs(np.diff(masses))) != 0):
                    raise ValueError('False sibling galaxies (mebership masking not working properly)')
            
            duplMassMask = np.array([masses[0] for masses in duplMasses]) > 1e14
            uniqueSteps = uniqueSteps[duplMassMask]
            galMask = galMask[duplMassMask]
            print('{} of {} duplicates kept after cluster mass cut'
                  .format(len(duplMassMask) - np.sum(duplMassMask == False), len(duplMassMask))) 
            
            haloGroups = [output_clusters.create_group('halo_{}'.format(nHalos+i))
                          for i in range(len(uniqueSteps))]
            print('created halo groups {}'.format([r.name for r in haloGroups]))
            nHalos += len(uniqueSteps)
        else:
            haloGroups = [output_clusters.create_group('halo_{}'.format(nHalos))]
            print('created halo group {}'.format(haloGroups[0].name))
            nHalos += 1

        # -----------------------------------------------------------------------------------
        # loop through each duplicated halo (only 1 loop in the case of no duplication)

        for j in range(len(galMask)):

            # create a halo-property and galaxy-property group in each halo_k group
            galPropGroup = haloGroups[j].create_group('galaxyProperties')
            haloPropGroup = haloGroups[j].create_group('haloProperties')

            host_quantityModifiers = {
                'hostHaloMass':'halo_mass',
                'hostIndex':'halo_index',
                'hostHaloTag':'fof_halo_tag',
                'step':'halo_step'
            }
            
            if(len(galMask) > 1): print('working on duplicate {} ({} gals)'
                                        .format(j+1, np.sum(galMask[j])))
            
            galProps = []
            protoDC2g.visit(galProps.append)
            galProps = np.array(galProps)
            hostPropMask = np.array([p in host_quantityModifiers.keys() for p in galProps])
            hostPropIdx = np.linspace(0, len(galProps)-1, len(galProps), dtype=int)[hostPropMask]
            hostProps = galProps[hostPropIdx]
            galProps = np.delete(galProps, hostPropIdx)

            # ------------------------------ GROUP ATTRIBUTES ------------------------------
            # save all halo-wide properties as group attributes
            for prop in hostProps:
                data = protoDC2g[prop][:][galMask[j]]
                if( sum(abs(np.diff(data))) != 0):
                    raise ValueError('False sibling galaxies (mebership masking not working properly)')
                modifiedProp = host_quantityModifiers[prop]
                # -- OLD HALO ATTRIBUTES -- haloGroups[j].attrs.create(modifiedProp, data[0])
                haloPropGroup.create_dataset(modifiedProp, data=data[0])
            
            # calculate some new halo properties not present in the catalog
            centralGal_mask = np.array(protoDC2g['isCentral'][:][galMask[j]], dtype=bool)
            centralGal_index = np.r_[0:np.sum(galMask):1][centralGal_mask]
            if(len(centralGal_index) != 1): 
                print('more than one central in this halo!')
                galRAs = protoDC2g['ra'][:][galMask[j]]
                hostRA = np.median(galRAs)[0]
                galDecs = protoDC2g['dec'][:][galMask[j]]
                hostDec = np.median(galDecs)[0]
            else:
                hostRA = protoDC2g['ra'][:][galMask[j]][centralGal_index][0]
                hostDec = protoDC2g['dec'][:][galMask[j]][centralGal_index][0]
            galZ = protoDC2g['redshift'][:][galMask[j]]
            hostZ = np.median(galZ)
            hostZErr = 1.25 * (np.std(galZ) / np.sqrt(len(galZ)))
            pecV = ct.LOS_properVelocity(galZ, hostZ)
            galDisp_obs = stat.bDispersion(pecV)[0]
            galDisp_3d = stat.dmDispersion(protoDC2g['vx'][:][galMask[j]], 
                                           protoDC2g['vy'][:][galMask[j]], 
                                           protoDC2g['vz'][:][galMask[j]])
            
            # -- OLD HALO ATTRIBUTES --
            #haloGroups[j].attrs.create('halo_ra', hostRA)
            #haloGroups[j].attrs.create('halo_dec', hostDec)
            #haloGroups[j].attrs.create('halo_z', hostZ)
            #haloGroups[j].attrs.create('halo_z_err', hostZErr)
            #haloGroups[j].attrs.create('gal_vel_disp_obs', galDisp_obs)
            #haloGroups[j].attrs.create('gal_vel_disp_1d', galDisp_1d)
            haloPropGroup.create_dataset('halo_ra', data = hostRA)
            haloPropGroup.create_dataset('halo_dec', data = hostDec)
            haloPropGroup.create_dataset('halo_z', data = hostZ)
            haloPropGroup.create_dataset('halo_z_err', data = hostZErr)
            haloPropGroup.create_dataset('gal_vel_disp_obs', data = galDisp_obs)
            haloPropGroup.create_dataset('gal_vel_disp', data = galDisp_3d)

            # save all sod properties of the halo from the haloCatalog as attrbiutes
            # -- OLD HALO ATTRIBUTES -- sodStepIdx = np.where(haloSteps == haloGroups[j].attrs['halo_step'])[0][0]
            sodStepIdx = np.where(haloSteps == haloPropGroup['halo_step'])[0][0]
            sodTags = gio.gio_read(haloCatalog[sodStepIdx], 'fof_halo_tag')
            sodIdx = np.where(sodTags == halos[k])
            
            for prop in sodProps:
                data = gio.gio_read(haloCatalog[sodStepIdx], prop)[sodIdx][0]
                # -- OLD HALO ATTRIBUTES -- haloGroups[j].attrs.create(prop, data)
                haloPropGroup.create_dataset(prop, data=data)
            print('saved all halo attributes')

            # ------------------------------ GROUP DATASETS ------------------------------
            # save all protoDC2 galaxy data columns for the current halo's member galaxies
            galProps_objType = np.array([isinstance(protoDC2g[key], h5py.Group) for key in galProps])
            gal_groups = galProps[galProps_objType]
            gal_datasets = galProps[~galProps_objType]
            for g in range(len(gal_groups)):
                group = gal_groups[g]
                # -- OLD HALO ATTRIBUTES -- haloGroups[j].create_group(group)
                galPropGroup.create_group(group)
            for d in range(len(gal_datasets)):
                dset = gal_datasets[d]
                data = protoDC2g[dset][:][galMask[j]]
                # -- OLD HALO ATTRIBUTES -- haloGroups[j].create_dataset(dset, data=data)
                galPropGroup.create_dataset(dset, data=data)
                # -- OLD HALO ATTRIBUTES -- if(len(haloGroups[j][dset]) != np.sum(galMask[j])):
                if(len(galPropGroup[dset]) != np.sum(galMask[j])):
                    raise ValueError('galaxy population size not the same across columns (bad masks)')
            print('saved all galaxy datasets')
 
    print('Done. Took {:.2f} s'.format(time.time() - start))
    return 0


# ====================================================================================================


def makeCatalog_steps():
    '''
    This function takes the protoDC2 step-wise galaxy catalog and reformats it as a cluster catalog, 
    structured as an hdf5 file as follows: 

    - The attributes of the root group '/' give information on the cosmology of the simulation
    - The root group has a child group for each halo of mass > 1e14, the names being sequential labels. 
      For example, the tenth halo located by the code is found at '/halo_10' in the hdf file
    - Each halo group contains x datasets which correspond to the data columns of the original 
      protoDC2 catalog, with consistent column names, for all of that halo's member galaxies. If a 
      halo is the host to 120 galaxies, then each of the x data columns are 120 elements long.
    - A few columns from the original catalog are missing from the halo group datasets. This is because
      the galaxy catalog has a few columns that apply halo-wide. For instance, the hostHaloMass is the 
      same value for every galaxy in the halo. Rather than repeating the information for every member
      galaxy, halo-wide properties are saved as group attributes. In addition to this data, the SO 
      properties of each halo are also saved as attributes, as well as a few other quantities 
      calculated in this code. Each halo group has the following attributes:
      
      From protoDC2:
        -   hostHaloMass (equivalent to the merger-tree halo mass (FOF mass minus fragments))
        -   hostHaloTag  (the FOF tag)
        -   hostIndex
        -   step
      From the SO catalog:
        -   sod_halo_mass      (m200)
        -   sod_halo_radius    (r200)
        -   sod_halo_vel_disp  (km/s)
        -   sod_halo_ke
        -   sod_halo_cdelta
        -   sod_halo_cdelta_error
        -   sod_halo_c_acc_mass
      Calculated here:
        -   Nothing is calculated here as in the makeCatalog() function, as there is no "observer"
            in the step catalog, and all galaxies are found at the same cosmological redshift (the step)
    Finally - this code as it stands should be run on datastar
    '''

# ====================================================================================================

    start = time.time()
    
    # load halo sod catalog
    haloCatalogPath = '/media/luna1/dkorytov/data/AlphaQ/sod'
    haloCatalog = glob.glob('{}/*.sodproperties'.format(haloCatalogPath))
    haloSteps = np.array([int(f.split('.')[0].split('-')[-1]) for f in haloCatalog])

    # define desired sod halo properties
    sodProps = ['sod_halo_radius', 'sod_halo_mass', 'sod_halo_ke', 'sod_halo_vel_disp', 
                'sod_halo_cdelta', 'sod_halo_cdelta_error', 'sod_halo_c_acc_mass']
    
    # load protoDC2 catalog
    #protoDC2Path = '/media/luna1/dkorytov/projects/protoDC2/output/mock_B_full_nocut.hdf5'
    protoDC2Path = '/media/luna1/dkorytov/projects/protoDC2/output/snapshot_box/mock_nocut_A.hdf5'
    #path_out = 'data/protoDC2_catalog_pecZ.hdf5'
    path_out = 'data/protoDC2_STEPS_clusters_nocut_A.hdf5'
    protoDC2 = h5py.File(protoDC2Path, 'r')
    outputFile = h5py.File(path_out, 'w')
    haloNum = 0

    # copy global attributes from original catalog
    # (as of right now, this does nothing, the step-wise catalog does not contain the cosmology 
    # of the simulation as attributes)
    globalAttrs = list(protoDC2.attrs.keys())
    for attr in globalAttrs:
        outputFile.attrs.create(attr, protoDC2.attrs[attr])

    # ===========================================================================================
    # ============================ loop through each simulation step ============================

    steps = list(protoDC2.keys())

    for i in range(len(steps)):
        
        step = int(steps[i])
        stepGals = protoDC2[steps[i]]
        h = 0.702
        
        print('\n------------ working on STEP {} ({}/{}) ------------'.format(step, i+1, len(steps)))

        # find all unique halo tags in protoDC2
        massMask = (stepGals['hostHaloMass'][:] / h) > 1e14
        uniqueTags = np.unique(stepGals['hostHaloTag'][:][massMask], return_counts = True)
        halos = uniqueTags[0]
        nGals = uniqueTags[1]
        duplicates=0

        # ---------------------------------------------------------------------------------------

        # loop through all halos found in the current step
        for k in range(len(halos)):

            print('\nworking on halo {} ({}/{})'.format(halos[k], k+1, len(halos)))

            # boolean mask of current halo galaxy membership
            galMask = stepGals['hostHaloTag'][:] == halos[k]
            # makeCatalog() checks for lightcone halo duplication here, which of course is not needed
            haloNum += 1
            haloGroup = outputFile.create_group('halo_{}'.format(haloNum))
            print('created hdf group at {}'.format(haloGroup.name))

            # ------------------------------ GROUP ATTRIBUTES ------------------------------
            
            host_quantityModifiers = {
                'hostHaloMass':'host_halo_mass',
                'hostIndex':'halo_index',
                'hostHaloTag':'fof_halo_tag',
            }
            
            galProps = np.array(list(stepGals.keys()))
            hostPropMask = np.array([p in host_quantityModifiers.keys() for p in galProps])
            hostPropIdx = np.linspace(0, len(galProps)-1, len(galProps), dtype=int)[hostPropMask]
            hostProps = galProps[hostPropIdx]
            galProps = np.delete(galProps, hostPropIdx)
            
            # save all halo-wide properties as group attributes
            for prop in hostProps:
                data = stepGals[prop][:][galMask]
                if( sum(abs(np.diff(data))) != 0):
                    raise ValueError('False sibling galaxies (mebership masking not working properly)')
                modifiedProp = host_quantityModifiers[prop]
                haloGroup.attrs.create(modifiedProp, data[0])
            haloGroup.attrs['host_halo_mass'] = haloGroup.attrs['host_halo_mass'] / h
           
            # save step number as group attribute rather than root group name
            haloGroup.attrs.create('halo_step', step)

            # save all sod properties of the halo from the haloCatalog as attrbiutes
            sodStepIdx = np.where(haloSteps == step)[0][0]
            sodTags = gio.gio_read(haloCatalog[sodStepIdx], 'fof_halo_tag')
            sodIdx = np.where(sodTags == halos[k])
            
            for prop in sodProps:
                data = gio.gio_read(haloCatalog[sodStepIdx], prop)[sodIdx]
                haloGroup.attrs.create(prop, data)
            print('saved all halo attributes')
            
            # ------------------------------ GROUP DATASETS ------------------------------
            # save all protoDC2 galaxy data columns for the current halo's member galaxies
            for p in range(len(galProps)):
                prop = galProps[p]
                data = stepGals[prop][:][galMask]
                haloGroup.create_dataset(prop, data=data)
                if(len(haloGroup[prop]) != np.sum(galMask)):
                    raise ValueError('galaxy population size not the same across columns (bad masks)')
            print('saved all galaxy datasets')
           
    print('Done. Took {:.2f} s'.format(time.time() - start))
    return 0


# ====================================================================================================


if __name__ == '__main__':
    makeCatalog()
