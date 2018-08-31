# Joe Hollowed
# HEP 6/21/17

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import pdb
import glob
import numpy as np
import clstrTools as ct
from astropy.io import fits
from numpy.lib.recfunctions import append_fields

def assignTypes():
    # Function to assign galaxy spectral types to SDSS galaxies based on emission line criteria as
    # given in Bayliss+ 2016

    # find clusters in redmapper data that correspond to queried spec datasets
    galsPath = '../../data/sdss/sdss_spec_catalog'
    redMapper = '../../data/sdss/redMapper_catalog.fits'

    galsFiles = sorted(glob.glob('{}/*galaxies.npy'.format(galsPath)))
    ids = [g.split('_')[-2].split('/')[-1] for g in galsFiles]
        
    all_clusters = fits.open(redMapper)[1].data
    cluster_indices = np.array([np.where(all_clusters['NAME'] == i)[0][0] for i in ids])
    clusters = all_clusters[cluster_indices]
    
    for j in range(len(clusters)):
        # classify galaxy spectral type for each cluster
        cluster = clusters[j]
        gals = np.load(galsFiles[j])
        types = np.zeros(len(gals), dtype='U32')

        oii_w = gals['oii_3726_eqw']
        oii_SN = abs(gals['oii_3726_flux']/gals['oii_3726_flux_err'])
        hd_w = gals['h_delta_eqw']
        hd_SN = abs(gals['h_delta_flux']/gals['h_delta_flux_err'])
        gal_ids = gals['bestObjID']

        for i in range(len(gals)):
            if(oii_w[i] == 0 or oii_SN[i] <= 2):
                # passive or PSF
                if(hd_w[i]) < 3: types[i] = 'k_'
                elif(hd_w[i] >=3 and hd_w[i] <=8): types[i] = 'k+a_'
                elif(hd_w[i] > 8): types[i] = 'a+k_'
            else:
                # star-forming
                if(oii_w[i] > -40 and hd_w[i] < 4): types[i] = 'e(c)_'
                elif(oii_w[i] <= -40): types[i] = 'e(b)_'
                elif(oii_SN[i] > 2): types[i] = 'e(a)_'
   
        # append a new 'type' column to the 'gals' numpy rec array, and save
        gals = append_fields(gals, 'gal_type', types, usemask = False, asrecarray = True)
        np.save('{}'.format(galsFiles[j]), gals)
        if(j%20==0):
            print('Saved data for cluster {} ({}/{})'.format(cluster['NAME'], j, len(clusters)))
                
            

        

    

