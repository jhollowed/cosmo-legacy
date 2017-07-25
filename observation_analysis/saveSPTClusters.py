'''
Joe Hollowed
HEP 6/2017
'''
import pdb
import numpy as np
import clstrTools as ct
import interlopers as lop
import massConversion as mc
import astropy.io.fits as fits
import dispersionStats as stat

def readClusters():
   
    # load data
    print('loading data') 
    dataPath = '/home/joe/skydrive/Work/HEP/data/spt'
    saveDest = '{}/sptgmos_clusters'.format(dataPath)
    sptgmos_file = '{}/sptgmos_clusters.fits'.format(dataPath)
    sptsz_file = '{}/SPT-SZ.fits'.format(dataPath)
    sptgmos = fits.open(sptgmos_file)[1:]
    sptsz = fits.open(sptsz_file)[1].data
    ids = [sptgmos[i].header['EXTNAME'] for i in range(len(sptgmos))]
    skipped_ids = []
    
    # consider each SPT cluster found in the spt_gmos sample
    for j in range(len(ids)):
        
        # find this cluster's position in the SZ-catalog
        host = sptgmos[j]
        host_id = host.header['EXTNAME']
        if(host_id == 'SPT-CLJ0539-5744'): host_id = 'SPT-CLJ0540-5744'
        sz_index = np.where(sptsz['SPT_ID'] == host_id)
        print('\n\nWorking on cluster {} ({}/{})'.format(host_id, j+1, len(ids)))
        
        # load galaxy data
        host = sptsz[sz_index] 
        gals = sptgmos[j].data
        pdb.set_trace()
        print('\ncluster has {} gals'.format(len(gals)))
 
        # remove interlopers, find cluster redshift and galaxy velocities
        print('removing interlopers')
        if(len(gals)) < 15: binZ = False
        else: binZ = True
        [gal_vs, gal_vsErr, host_z, host_zErr, mask] = \
                                        lop.sig_interlopers(gals['Slitz'], gals['Slitzerr'], binZ=binZ)
        print('{} interlopers found'.format(len(mask[~mask])))

        # organize host cluster data (get dispersion, mass, radius, etc.)
        print('measuring cluster properties')
        host_vDisp = stat.bDispersion(gal_vs[mask])
        host_zErr = stat.bootstrap_bAverage_err(gals['Slitz'][mask])
        host_vDispErr = stat.bootstrap_bDispersion_err(gal_vs[mask])
        host_nMem = len(gal_vs[mask])
    
        try:
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
        
        print('dispersion:{:.2f}, z:{:.2f}, sz_z:{:.2f}, m200:{:.2f}, r200:{:.2f}'
              .format(host_vDisp, host_z, host['REDSHIFT'][0], host_m200, host_r200))

        # define column names for final data outputs
        print('saving data')
        host_cols = ['spt_id', 'spec_z', 'spec_zErr', 'v_disp', 'v_dispErr', 'n_mem', 'sz_z', 
                     'sz_zErr', 'RA', 'Dec', 'm200', 'r200']
        gal_cols = ['RA', 'Dec', 'MFlag', 'z', 'zErr', 'v', 'VErr', 'EW_OII', 'sig(EW_OII)', 'EW_Hd', 
                    'sig(EW_Hd)', 'd4000', 'sig(d4000)', 'mag_r', 'mag_i', 'm-m*(r)', 'm-m*(i)', 
                    'gal_type', 'mem?']

        # save data
        host_data = [host_id, host_z, host_zErr, host_vDisp, host_vDispErr, host_nMem, 
                     host['REDSHIFT'][0], host['REDSHIFT_UNC'][0], host['RA'][0], host['DEC'][0], 
                     host_m200, host_r200]

        host_output = np.rec.fromarrays(host_data, names=host_cols)
        gals_output = np.rec.fromarrays([gals['SlitcRA'], gals['SlitDec'], gals['MFlag'], gals['Slitz'],
                                         gals['Slitzerr'], gal_vs, gal_vsErr, gals['EW_OII'], 
                                         gals['sig(EW_OII)'], gals['EW_Hd'], gals['sig(EW_Hd)'], 
                                         gals['d4000'], gals['sig(d4000)'], gals['mag_r'], 
                                         gals['mag_i'], gals['m-m*(r)'], gals['m-m*(i)'], 
                                         gals['gal_type'], mask], names=gal_cols)
        
        np.save('{}/{}_prop.npy'.format(saveDest, host_id), host_output)
        np.save('{}/{}_galaxies.npy'.format(saveDest, host_id), gals_output)
        print('done')
    print('\n\n skipped Ids: {}'.format(skipped_ids))
