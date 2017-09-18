'''
Joe Hollowed 
COSMO-HEP 2017
'''

import pdb
import glob
import h5py
import numpy as np
from dtk import gio
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


def makeCatalog():
    '''
    This function takes the protoDC2 galaxy catalog and reformats it as a cluster catalog, structured 
    as an hdf5 file as follows: 

    - The attributes of the root group '/' give information on the cosmology of the simulation
    - The root group had a child group for each halo of mass > 1e14, the names of which being the 
      halo's fof tag. For example, the halo with tag 123456 is found at '/123456' in the hdf file
    - If duplicates of a halo were found (or at least, the same tag was used again in a later snapshot),
      then the step number of the halo is appended to the end of the group name. For example, if 
      halo with tag 456789 was found twice at step 286 and 247, the two hdf groups are created as 
      '/456789-286' and '/456789-247'
    - Each halo group contains 97 datasets which correspond to the data columns of the original 
      protoDC2 catalog, with consistent column names, for all of that halo's member galaxies. If a 
      halo is the host to 120 galaxies, then each of the 97 data columns are 120 elements long.
    - A few columns from the original catalog are missing from the halo group datasets. This is because
      the galaxy catalog has a few columns that apply halo-wide. For instance, the hostHaloMass is the 
      same value for every galaxy in the halo. Rather than repeating the information for every member
      galaxy, halo-wide properties are saved as group attributes. In addition to this data, the SO 
      properties of each halo are also saved as attributes, as well as a few other quantities 
      calculated in this code. Each halo group has the following attributes:
      
      From protoDC2:
        -   hostHaloMass (equivalent to the FOF mass)
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
        -   RA   (median of meber galaxies RA)
        -   Dec  (median of member galaxies declination)
        -   z    (median of member galaxies z's)
        -   zErr (standard error of the median - 1.25 * SDOM)
        ** If a more robust statistic turns out to be needed for the last three quanties above, or the 
           redshift error ought to be simulated rather than assumed from a normal distribution, then I 
           will make those changes. But nearly all halos in this catalog have large numbers of member 
           galaxies (on order 10^3) **

    Finally - this code as it stands should be run on datastar
    '''

# ====================================================================================================

    start = time.time()
    
    # load halo sod catalog
    haloCatalogPath = '/media/luna1/dkorytov/data/AlphaQ/sod'
    haloCatalog = glob.glob('{}/*.sodproperties'.format(haloCatalogPath))
    haloSteps = np.array([int(f.split('.')[0].split('-')[-1]) for f in haloCatalog])

    # load protoDC2 catalog
    protoDC2Path = '/media/luna1/dkorytov/projects/protoDC2/output/mock_shear_full.hdf5'
    protoDC2 = h5py.File(protoDC2Path, 'r')
    path_out = 'data/protoDC2_catalog.hdf5'
    outputFile = h5py.File(path_out, 'w')

    # copy global attributes from original catalog
    globalAttrs = list(protoDC2.attrs.keys())
    for attr in globalAttrs:
        outputFile.attrs.create(attr, protoDC2.attrs[attr])

    # find all unique halo tags in protoDC2
    massMask = protoDC2['hostHaloMass'][:] > 1e14 
    uniqueTags = np.unique(protoDC2['hostHaloTag'][:][massMask], return_counts = True)
    halos = uniqueTags[0]
    nGals = uniqueTags[1]
    duplicates=0


    # ---------------------------------------------------------------------------------------

    # loop through all halos
    for k in range(len(halos)):
        print('\nworking on halo {} ({}/{})'.format(halos[k], k+1, len(halos)))

        # boolean mask of current halo galaxy membership
        galMask = [protoDC2['hostHaloTag'][:] == halos[k]]



        # -----------------------------------------------------------------------------------
        # check for duplication

        hostSteps = protoDC2['step'][:][galMask[0]] 
        if( sum(abs(np.diff(hostSteps))) != 0):
            duplicates += 1
            print('Found duplicate ({})'.format(duplicates))
            uniqueSteps = np.unique(hostSteps)
            galMask = np.array([np.logical_and(galMask[0], protoDC2['step'][:] == step) 
                       for step in uniqueSteps])
            
            # make sure all duplicates still fit the cluster mass criteria
            duplMasses = [protoDC2['hostHaloMass'][:][mask] for mask in galMask]
            
            for masses in duplMasses:
                if( sum(abs(np.diff(masses))) != 0):
                    raise ValueError('False sibling galaxies (mebership masking not working properly)')
            
            duplMassMask = np.array([masses[0] for masses in duplMasses]) > 1e14
            uniqueSteps = uniqueSteps[duplMassMask]
            galMask = galMask[duplMassMask]
            print('{} of {} duplicates kept after cluster mass cut'
                  .format(len(duplMassMask) - np.sum(duplMassMask == False), len(duplMassMask))) 

            haloGroups = [outputFile.create_group('{}-{}'.format(halos[k], step)) 
                          for step in uniqueSteps]
        else:
            haloGroups = [outputFile.create_group('{}'.format(halos[k]))]


        
        # -----------------------------------------------------------------------------------
        # loop through each duplicated halo (only 1 loop in the case of no duplication)

        for j in range(len(galMask)):
            if(len(galMask) > 1): print('working on duplicate {} ({} gals)'
                                        .format(j+1, np.sum(galMask[j])))
            galProps = np.array(list(protoDC2.keys()))
            hostPropIdx = np.where( (['host' in p for p in galProps]) | (galProps == 'step') )[0]
            hostProps = galProps[hostPropIdx]
            galProps = np.delete(galProps, hostPropIdx)

            # ------------------------------ GROUP ATTRIBUTES ------------------------------
            # save all halo-wide properties as group attributes
            for prop in hostProps:
                data = protoDC2[prop][:][galMask[j]]
                if( sum(abs(np.diff(data))) != 0):
                    raise ValueError('False sibling galaxies (mebership masking not working properly)')
                haloGroups[j].attrs.create(prop, data[0])
            
            # calculate some new halo properties not present in the catalog
            galRA = protoDC2['ra'][:][galMask[j]]
            galDec = protoDC2['dec'][:][galMask[j]]
            galZ = protoDC2['redshift'][:][galMask[j]]
            medianZErr = 1.25 * (np.std(galZ) / np.sqrt(len(galZ)))
            haloGroups[j].attrs.create('ra', np.median(galRA))
            haloGroups[j].attrs.create('dec', np.median(galDec))
            haloGroups[j].attrs.create('z', np.median(galZ))
            haloGroups[j].attrs.create('zErr', medianZErr)

            # save all sod properties of the halo from the haloCatalog as attrbiutes
            sodStepIdx = np.where(haloSteps == haloGroups[j].attrs['step'])[0][0]
            sodTags = gio.gio_read(haloCatalog[sodStepIdx], 'fof_halo_tag')
            sodIdx = np.where(sodTags == halos[k])
            sodProps = ['sod_halo_radius', 'sod_halo_mass', 'sod_halo_ke', 'sod_halo_vel_disp', 
                        'sod_halo_cdelta', 'sod_halo_cdelta_error', 'sod_halo_c_acc_mass']
            for prop in sodProps:
                data = gio.gio_read(haloCatalog[sodStepIdx], prop)[sodIdx]
                haloGroups[j].attrs.create(prop, data)
            print('saved all halo attributes')

            # ------------------------------ GROUP DATASETS ------------------------------
            for p in range(len(galProps)):
                prop = galProps[p]
                data = protoDC2[prop][:][galMask[j]]
                haloGroups[j].create_dataset(prop, data=data)
                if(len(haloGroups[j][prop]) != np.sum(galMask[j])):
                    raise ValueError('galaxy population size not the same across columns (bad masks)')
            print('saved all galaxy datasets')
    
    print('Done. Took {:.2f} s'.format(time.time() - start))
    return 0



# ====================================================================================================


if __name__ == '__main__':
    makeCatalog()
