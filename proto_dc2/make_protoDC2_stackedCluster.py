'''
Joe Hollowed
COSMO-HEP 2017
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import pdb
import h5py
import time
import numpy as np
import clstrTools as ct

def stackGalaxies():
    
    start = time.time()

    clusterCatalog = h5py.File('data/protoDC2_clusters_shear_nocut_A.hdf5', 'r')
    stackedCatalog = h5py.File('data/protoDC2_haloStack_shear_nocut_A.hdf5', 'w')
    clusters = list(clusterCatalog.keys())

    for j in range(len(clusters)):
       
        halo = clusterCatalog[clusters[j]]
        print('working on halo {} ({}/{})'.format(halo.attrs['fof_halo_tag'], j, len(clusters)))

        # get cluster properties
        zHost = halo.attrs['halo_z']
        zHost_err = halo.attrs['halo_z_err']
        aHost = 1/(1+zHost)
        sodDisp = halo.attrs['sod_halo_vel_disp'] * aHost
        hostRadius = halo.attrs['sod_halo_radius']
        hostCoords = np.array([halo.attrs['halo_ra'], halo.attrs['halo_dec']]) / 3600

        # get LOS peculiar velocities for all member galaxies using observed redshift
        # and projected radial distances 
        print('projecting distances')
        z = halo['redshiftObserver'][:]
        pecV = ct.LOS_properVelocity(z, zHost)
        galCoords = np.array([halo['ra'][:], halo['dec'][:]]).T / 3600
        galDistances = ct.projectedDist(galCoords, hostCoords, zHost, dist_type='comoving')
       
        # normalize peculiar velocity of each cluster member by this clusters 
        # particle-based dispersion, and each radial distance by this cluster's r200. 
        # The particle-based dispersion is used here so that this value is insensitive 
        # to the statistics used to find the galaxy-based dispersion (details of estimator 
        # matter more since sample sizes are far smaller than those of particles)
        print('normalizing quantities')
        pecV_norm = (pecV / sodDisp)
        galDistances_norm = (galDistances / hostRadius)
        newData = [pecV_norm, galDistances_norm]
        newColumns = ['pecV_normed', 'projDist_normed']

        print('appendnig data to stack columns')
        for column in list(halo.keys()):
            data = halo[column]
            if(j == 0):
                stackedCatalog.create_dataset(column, (len(data),), maxshape=(None,), data=data)
            else:
                currSize = stackedCatalog[column].shape[0]
                incrSize = len(data)
                newSize = currSize + incrSize
                stackedCatalog[column].resize((currSize + incrSize,))
                stackedCatalog[column][currSize:newSize] = data
        
        for i in range(len(newColumns)):
            column = newColumns[i]
            data = newData[i]
            if(j == 0):
                stackedCatalog.create_dataset(column, (len(data),), maxshape=(None,), data=data)
            else:
                currSize = stackedCatalog[column].shape[0]
                incrSize = len(data)
                newSize = currSize + incrSize
                stackedCatalog[column].resize((currSize + incrSize,))
                stackedCatalog[column][currSize:newSize] = data
    
    print(time.time()-start)
if __name__ == '__main__':
    stackGalaxies()
