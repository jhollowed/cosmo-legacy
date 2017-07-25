'''
Joe Hollowed
HEP 2017

This script adds SZ data to all 'prop.npy' files for each SPT cluster, including the SZ measurement
of redshift, the SZ position in RA,DEC, and the SZ-derived mass in M500, which is converted to M200.

'''

import pdb
import glob
import numpy as np
import numpy.lib.recfunctions
import astropy.io.fits as fits
from clstrTools import convertMass as toM200
from clstrTools import mass_to_radius as toR200  
 
# Gather data
sptPath = '/home/joe/skydrive/Work/HEP/data/spt'
clusterPath = '{}/sptgmos_lit_clusters'.format(sptPath)
szPath = '{}/SPT-SZ.fits'.format(sptPath)

sz = fits.open(szPath)[1].data
clusterFiles = glob.glob('{}/SPT*prop*'.format(clusterPath))
clusterTags = np.array([f.split('_')[-2].split('/')[-1] for f in clusterFiles])

# Mask SZ catalog to only keep clusters that are present in the SPTGMOS-Lit spec catalog
mask = np.array([tag in clusterTags for tag in sz['SPT_ID']])
sz = sz[mask]
szTags = sz['SPT_ID']

# Add new data columns to each GMOS cluster's .npy file from the SZ catalog
for n in range(len(clusterFiles)):

    # Rename columns (I didn't like how I originally made them)
    f = clusterFiles[n]
    tag = clusterTags[n]
    cluster = np.load(f)
    cluster.dtype.names = ('spt_id', 'spec_z', 'v_disp', 'n_mem')
    
    # If current cluster is not in the SZ catalog for any reason, skip it
    mask = szTags == tag
    if sum(mask) == 0: 
        print('Cluster {} not found in SZ Catalog; skipping'.format(tag))
        np.save(f, cluster)
        continue
    
    # Convert SZ derived M500c to M200c, and find R200c
    sz_cluster = sz[mask]
    z = sz_cluster['REDSHIFT']
    mass = toM200(sz_cluster['M500'][0], 500, z[0], h=70)
    rad = toR200(mass, z[0], mdef='200c')

    # Add new data columns
    newCols = ['sz_z', 'RA', 'Dec', 'm200', 'r200']
    newData = [z[0], sz_cluster['RA'][0], sz_cluster['DEC'][0], mass, rad]
    newTypes = [z.dtype,  sz_cluster['RA'].dtype, sz_cluster['DEC'].dtype, sz_cluster['M500'].dtype, 
                sz_cluster['M500'].dtype]
    
    cluster = np.atleast_1d(cluster)
    for i in range(len(newData)):
        cluster = np.lib.recfunctions.append_fields(cluster, newCols[i], [newData[i]], 
                                                    dtypes=newTypes[i], usemask=False, 
                                                    asrecarray=True)
    print('Added new data to cluster {} ({}/{})'.format(tag, n, len(clusterFiles)))
    np.save(f, cluster)
