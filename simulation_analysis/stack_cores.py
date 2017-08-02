'''
Joe Hollowed
Last edited 2/7/2017

Script to stack large cluster-sized dark matter halo cores. This means to normalize each 
of the cores radius and velocities by properties of its host cluster. The following data 
saved per core in the stacked (or ensemble) halo:

- The radial position: vec{r_core} - vec{r_halo_mean} where r is a cartesian 
	position vector. This radial position is multiplied by the scale
	factor, a, given by the host halo's redshift, and is therefore in physical coordinates. 
- The normalized radial position: the radial position described above, scaled by 
	the inverse sod halo radius (r200) of the host halo
- The radial velocity: ((v_core - v_halo_mean) {DOT} (r / |r| )
	where r is the radial position vector of the core as described above. This radial
	velocity is then multiplied by the scale factor, a, given by
	the host halo's redshift.
- The radial "coreNorm" and "dmNorm"  velocity: the radial velocity above, 
	scaled by the inverse of the core velocity dispersion and dark matter (particle) 
	velocity dispersion of the host cluster, respectively.
- The halo-respective cartesian velocity: simply the magnitude of the relative velocity vector
	between the core and the halo center. This is denoted by v_norm (note the lack of the
	'_rad_' specifier)
- The 2d radial position: the radial position projected onto the y-z plane, so the x-axis, 
	which will be chosen as our 1-dimensional velocity, is along the "line of sight"
- The normalized 2d radial position: as described above for the 3d case
- The 1d velocity of each core:  The relative 3d velocity projected onto the x-axis 
- The "coreNorm" 1d velocity: as described for the radial case above
- Host halo tag and core tag
- Core infall mass
- Core infall step
- Core infall halo tag
- Core radius

This script makes such a stacked halo at each redshift where there is at least one halo

'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import dtk
import pdb
import h5py
import glob
import numpy as np
import coreTools as ct
from scipy.stats import itemfreq
import numpy.lib.recfunctions as rcfn
from numpy.core import umath_tests as npm
from astropy.cosmology import WMAP9 as cosmo

def stackCores(catalog = 0, processed = True):

    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    processSuffix = ['un', ''][processed]
    corePath = '/home/jphollowed/data/hacc/alphaQ/coreCatalog/{}'.format(catalogName)
    stackFile = h5py.File('{}/stackedCores_{}processed.hdf5'.format(corePath, processSuffix), 'w')
    print('Read data from {} catalog and created file for stacking'.format(catalogName))

    zTool = dtk.StepZ(200, 0, 500)
    boxL = 256

    allHalos = h5py.File('{}/haloCores_{}processed.hdf5'.format(corePath, processSuffix), 'r')
    halo_steps = [int(step.split('_')[-1]) for step in list(allHalos.keys())]
    halo_zs = np.array([zTool.get_z(step) for step in halo_steps])
    print('Found all redshift groups in {} hdf5 file\nBeginning stacking...'.format(catalogName))


    for n in range(len(halo_zs)):
        # organize data for next step, and create new hdf5 group in stack file
        step = halo_steps[n]
        z = halo_zs[n]
        a = cosmo.scale_factor(z) 
        nextHalos = allHalos['step_{}'.format(step)]
        haloTags = list(nextHalos.keys())
        nextZGroup = stackFile.create_group('step_{}'.format(step))
        nextZGroup.attrs.create('z', z)
        print('\n{} halos at redshift {:.2f}'.format(len(haloTags), z))

        # find total number of cores that will be in this step's stack
        coreTot = sum([len(nextHalos[tag]['core_tag']) for tag in haloTags])

        print('creating datasets')
        # gather data that I already have in the un-stacked core files, faltten, and add to new hdf
        haveColumns = ['core_tag', 'infall_mass', 'infall_step', 'radius']
        for column in haveColumns:
            values = np.array([nextHalos[tag][column].value for tag in haloTags])
            if(len(values)>1): values = np.hstack(values)
            nextZGroup.create_dataset(column , data=values)

        # create datasets in new hdf for columns that will be computed below
        calcColumns = ['r', 'r_norm', 'v', 'v_coreNorm', 'v_dmNorm', 'v_rad', 
                       'v_rad_coreNorm', 'v_rad_dmNorm', 'v_tan', 'v_tan_coreNorm'
                       'v_tan_dmNorm', 'r_proj', 'r_proj_norm', 'v_los', 'v_los_coreNorm']
        for column in calcColumns:
            nextZGroup.create_dataset(column, shape = (coreTot,))
                            
        # keep track of index position as we put data from each halo into the file datasets
        startIndex = 0
        
        for j in range(len(haloTags)):
            halo = nextHalos[haloTags[j]].attrs
            cores = nextHalos[haloTags[j]]
            n_cores = len(cores['core_tag'])
            tag = haloTags[j]
            endIndex = startIndex + n_cores


            if(j%50 == 0):	
                    print('working on halo {} -- {} cores'.format(tag, n_cores))
            
            #if(tag == 709670038):
            #        # skip weird quintuplet halo 
            #        continue

            #----------------------------- 3 DIMENSIONAL ------------------------------------------	
            
            # gather halo velocity/position, and core velocities/positions, 
            # in form [vx, xy, vz]
            # Note: h5py datasets must be read in the form file['datasetName'][:] to yield 
            # an actual numpy array, file['datasetName'] will return an h5py object
            v_cores = np.array([cores['vx'][:], cores['vy'][:], cores['vz'][:]]).T
            v_halo = np.array([halo['sod_halo_mean_v{}'.format(p)] for p in ['x','y','z']])
            
            r_cores = np.array([cores['x'][:], cores['y'][:], cores['z'][:]]).T
            r_halo = np.array([halo['sod_halo_min_pot_{}'.format(p)] for p in ['x','y','z']])

            # Adjust positions of all cores that cross the periodic box bound, 
            # to reflect accurate distances. Factor of a to all resultant positions
            # to get physical distances
            r_cores = ct.unwrap_position(r_cores, r_halo)
            r_cores = r_cores * a
            r_halo = r_halo * a
            
            # Dot relative position with relative core velocity to get radial velocity
            # using the 'inner1d' row-wise dot function (it's fast)
            r_rel = r_cores - r_halo
            r_rel_mag = np.linalg.norm(r_rel, axis=1)
            r_rel_hat = np.divide(r_rel, np.array([r_rel_mag]).T)
            v_rel = v_cores - v_halo
            v_radial = npm.inner1d(v_rel, r_rel_hat)
            
            # save the magnitudes of the relative velocities
            v_rel_mag = np.linalg.norm(v_rel, axis=1)
                    
            # if v_radial, calculated above,  has nans, it is beacause the relative position 
            # between the halo minimum and a core is zero. In this case, any relative velocity 
            # is radial, so replace nan radial velocity with magnitude of relative velocity.
            centered_cores = np.isnan(v_radial)
            v_radial[centered_cores] = v_rel_mag[centered_cores]

            # Find tangential velocity. First obtain vector orthagonal to plane of motion, 
            # then cross that vector with the position vector r, normalize, and dot the result
            # with the relative velocity
            tan_dir = np.cross( np.cross(v_rel, r_rel, axisa=1), r_rel, axisa=1)
            tan_dir_mag = np.linalg.norm(tan_dir, axis=1)
            tan_dir_hat = np.divide(tan_dir, np.array([tan_dir_mag]).T)
            v_tan = npm.inner1d(v_rel, tan_dir_hat)
            # get rid of nans, in the case of cores that have pure radial velocity
            v_tan[np.isnan(v_tan)] = 0

            # check that calculations are consistent before continuing
            if( max(npm.inner1d(r_rel_hat, tan_dir_hat)) > 1e-6): 
                raise ValueError('tangent and radial unit vectors are not orthagonal for all cores')
            if( max(abs(np.linalg.norm(np.column_stack([v_radial, v_tan]), axis=1) 
                        - v_rel_mag)) > 1e-3):
                raise ValueError('velocity components do not superimpose to original velocity vector')
            
            # normalize cartesian and radial velocities and radii with respect to velocity 
            # dispersion and host halo
            r_norm = r_rel_mag / (halo['sod_halo_radius'])
            v_coreNorm = v_rel_mag / halo['core_vel_disp']
            v_dmNorm = v_rel_mag / halo['sod_halo_vel_disp']
            v_radial_coreNorm = v_radial / halo['core_vel_disp']
            v_radial_dmNorm = v_radial / halo['sod_halo_vel_disp']
            v_tan_coreNorm = v_tan / halo['core_vel_disp']
            v_tan_dmNorm = v_tan / halo['sod_halo_vel_disp']
        
            #----------------------------- 1 DIMENSIONAL ------------------------------------------	
            
            # repeat as above for 3d case
            v_1d = v_rel.T[0]
            r_cores_2d = np.array([r[1:] for r in r_cores])
            r_halo_2d = r_halo[1:]
            r_rel_2d = r_cores_2d - r_halo_2d
            r_rel_mag_2d = np.linalg.norm(r_rel_2d, axis=1)
    
            r_2d_norm = r_rel_mag_2d / (halo['sod_halo_radius'])
            v_1d_coreNorm = v_1d / halo['core_vel_disp']
    
            # --------------------------------------------------------------------------------------
            
            calcColumnData = [r_rel_mag, r_norm, v_rel_mag, v_coreNorm, v_dmNorm, v_radial, 
                              v_radial_coreNorm, v_radial_dmNorm, v_tan, v_tan_coreNorm, 
                              v_tan_dmNorm, r_rel_mag_2d, r_2d_norm, v_1d, v_1d_coreNorm]
            
            for k in range(len(calcColumns)):
                nextZGroup[calcColumns[k]][startIndex:endIndex] = calcColumnData[k]
    
            startIndex = endIndex

        print('Done')
