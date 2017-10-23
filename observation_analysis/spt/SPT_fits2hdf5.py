'''
Joe Hollowed
COSMO-HEP 10/2017
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'util'))

import pdb
import h5py
import numpy as np
import clstrTools as ct
import interlopers as lop
import massConversion as mc
import astropy.io.fits as fits
import dispersionStats as stat

def makehdf5():
    '''
    This function converts the .fits file containing the SPT-GMOS galaxy data to an
    hdf5 format. This simpy turns the fits tables into hdf5 groups. Then, each cluster
    within the original fits file is found in the SPT-SZ catalog, and it's cluster-wide
    properties are saved as group attributes in the new hdf5 file.
    '''
    print('loading data') 
    sptgmos = fits.open('data/sptgmos_clusters.fits')[1:]
    sptsz = fits.open('data/SPT-SZ.fits')[1].data
    outFile = h5py.File('data/sptgmos_clusters.hdf5', 'w')
    ids = [sptgmos[i].header['EXTNAME'] for i in range(len(sptgmos))]
    skipped_ids = []

    # loop over each cluster found in the spt_gmos sample
    for j in range(len(ids)):
        
        host = sptgmos[j]
        host_id = ids[j]
        # correct a cluster name which was given a slightly different sky position
        # when data was taken with GMOS, as opposed to the SPT-given position seen in
        # the SZ catalog
        if(host_id == 'SPT-CLJ0539-5744'): host_id = 'SPT-CLJ0540-5744'
        
        sz_index = np.where(sptsz['SPT_ID'] == host_id)
        print('\n\nWorking on cluster {} ({}/{})'.format(host_id, j+1, len(ids)))
        
        # load galaxy data
        host = sptsz[sz_index]
        gals = sptgmos[j].data
        print('\ncluster has {} gals'.format(len(gals)))
 
        # flag interlopers, find cluster redshift and galaxy velocities
        print('flagging interlopers')
        if(len(gals)) < 15: binZ = False
        else: binZ = True
        [gal_vs, gal_vsErr, host_z, host_zErr, mask] = lop.sig_interlopers(gals['Slitz'], 
                                                       gals['Slitzerr'])

        print('{} interlopers found'.format( len(gal_vs[~mask]) ))

        # organize host cluster data (get dispersion, mass, radius, etc.)
        print('finding cluster properties')
        host_vDisp = stat.bDispersion(gal_vs[mask])
        host_vDispErr = stat.bootstrap_bDispersion(gal_vs[mask])
        host_nMem = len(gal_vs[mask])
    
        try:
            # convert m500c to m200c masses. If more than 10 member galaxies, use the 
            # averaging-based cluster redshift. Otherwise, use the SZ redshift
            if(host_nMem >= 10):
                host_m200 = mc.MDelta_to_M200(host['M500'], 500, host_z)
                host_r200 = ct.mass_to_radius(host_m200, host_z, h=70)
            else:
                host_m200 = mc.MDelta_to_M200(host['M500'], 500, host['REDSHIFT'])
                host_r200 = ct.mass_to_radius(host_m200, host['REDSHIFT'], h=70)
        except RuntimeError:
            print('******** SKIPPING {} ************'.format(host_id))
            skipped_ids += [host_id]
            continue
       
        print('dispersion:{:.2f}, z:{:.2f}, sz_z:{:.2f}, m200:{:.2f} x 10^14, r200:{:.2f}'
              .format(host_vDisp, host_z, host['REDSHIFT'][0], host_m200/1e14, host_r200))

        # define column names for final data outputs
        print('saving data')
        host_cols = ['spt_id', 'spec_z', 'spec_zErr', 'v_disp', 'v_dispErr', 'n_mem', 'sz_z', 
                     'sz_zErr', 'RA', 'Dec', 'm200', 'r200']
        gal_cols = ['RA', 'Dec', 'MFlag', 'z', 'zErr', 'v', 'VErr', 'EW_OII', 'sig(EW_OII)', 
                    'EW_Hd', 'sig(EW_Hd)', 'd4000', 'sig(d4000)', 'mag_r', 'mag_i', 
                    'm-m*(r)', 'm-m*(i)', 'gal_type', 'member']
        host_data = [host_id, host_z, host_zErr, host_vDisp, host_vDispErr, host_nMem, 
                     host['REDSHIFT'][0], host['REDSHIFT_UNC'][0], host['RA'][0], host['DEC'][0], 
                     host_m200, host_r200]

        gal_data = [gals['SlitcRA'], gals['SlitDec'], gals['MFlag'], gals['Slitz'],
                    gals['Slitzerr'], gal_vs, gal_vsErr, gals['EW_OII'], 
                    gals['sig(EW_OII)'], gals['EW_Hd'], gals['sig(EW_Hd)'], 
                    gals['d4000'], gals['sig(d4000)'], gals['mag_r'], 
                    gals['mag_i'], gals['m-m*(r)'], gals['m-m*(i)'], 
                    gals['gal_type'], mask]
        
        # save data to hdf5
        hostGroup = outFile.create_group(host_id)
        for k in range(len(host_cols)):
            try: hostGroup.attrs.create(host_cols[k], host_data[k])
            except TypeError: hostGroup.attrs.create(host_cols[k], np.string_(host_data[k]))
        for k in range(len(gal_cols)):
            try: hostGroup.attrs.create(gal_cols[k], gal_data[k])
            except TypeError: hostGroup.attrs.create(gal_cols[k], np.string_(gal_data[k]))
        print('copied all data to hdf5')    
    
    print('Done.\n\n skipped Ids: {}'.format(skipped_ids))

if(__name__ == '__main__'):
    makehdf5()
