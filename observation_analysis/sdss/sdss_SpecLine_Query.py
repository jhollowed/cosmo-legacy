import os
import pdb
import time
import sqlcl
import numpy as np
from io import BytesIO
import clstrTools as ct
import astropy.io.fits as fits

def querySDSS(mask_richest = 0, richest_num = 100, start = 0, num = 0, r200_factor = 1):
    '''
    Query the SDSS data release for spectral objects in redmapper identified clusters
    :param mask_richest: boolean; whether or not to only query n richest clusters
    :param richest_num: int; how many clusters to use in the richness mask
    :param start: Cluster index to begin query at (if querying different clusters in different runs,
                  may have unexpected behavior if mask parameters are different between those runs)
    :param num: number of clusters to query, starting at 'start'
    :param r200_factor: multiples of r200 to use as query search radius
    :return: None. Save query results to numpy files.
    '''

    print('Reading fits')
    clustersFits = fits.open("../../data/sdss/redMapper_catalog.fits")
    saveDest = '../../data/sdss/sdss_spec_catalog/'
    # header list can be found in http://arxiv.org/pdf/1303.3562v2.pdf

    #data from fits
    clusters =  clustersFits[1].data

    #collect cluster richness and most likely bcg spec z's
    richness = clusters['lambda']

    #find 100 richest clusters
    print('Building masks')
    if(mask_richest == 1):
        richestIdx = np.argsort(-richness)[0:richest_num]
        allIdx = np.r_[0:len(richness)]
        rich_mask = np.array([r in richestIdx for r in allIdx], dtype=bool)
    else:
        rich_mask = np.ones(len(richness), dtype=bool)

    # add other masks to this 'and' if desired in the future
    # mask = np.bitwise_and(rich_mask) 
    mask = rich_mask
    clusters = clusters[mask]
    N = len(clusters)

    fails = []
    if num == 0: num = N-start
    if num > N: num = N-start
    spec_count = 0
    begin = time.time()
 
    # -- query columns are defined here:
    # goo.gl/gATHpL 
    # -- cluster (redMapper) data columns are defined here (pg.35) :
    # https://arxiv.org/abs/1303.3562
    
    # query for num clusters, starting at index start
    for i in range(start, num):
        try:
            clst = clusters[i]
            name = clst['NAME']
            ra = clst['RA']
            dec = clst['DEC']
            z = clst['Z_LAMBDA']
            zErr = clst['Z_LAMBDA_ERR']
            richness = clst['LAMBDA']
            richnessErr = clst['LAMBDA_ERR']
            numGal = richness / clst['S']
            numGal_err = richnessErr / clst['S']
            bcg_z = clst['Z_SPEC']
            bcg_ra = clst['RA_CEN']
            bcg_dec = clst['DEC_CEN']
            bcg_pcen = clst['P_CEN']
            ilum = clst['ILUM']
            rad = ct.richness_to_arcmin(richness, z, h=100, mdef='200m')
            search_radius = r200_factor * rad
            
            # Query and save spectral data (including galaxy info from SpecObjAll SDSS table and 
            # spectral line info from galSpecLine SDSS table) of all cluster objects 
            # (r200_factor * r200 arcmin search radius)
            print("\nQuerying spec objects within {:.2f} arcminutes of cluster {} (of {})"
                  .format(search_radius, i + 1, num))
            query_dtypes = ['i8'] + ['f8']*6 +  ['a32'] + ['f4']*12
            query_str = "select s.bestObjID, s.ra, s.dec, s.z, s.zErr, s.z_noqso, s.zErr_noqso, "\
                        "s.class, p.oii_3726_eqw, p.oii_3726_eqw_err, p.oii_3726_flux, "\
                        "p.oii_3726_flux_err, p.oii_3729_eqw, p.oii_3729_eqw_err, p.oii_3729_flux, "\
                        "p.oii_3729_flux_err, p.h_delta_eqw, p.h_delta_eqw_err, p.h_delta_flux, "\
                        "p.h_delta_flux_err from galSpecLine as p join "\
                        "specObjAll as s on p.specObjID = s.specObjID join "\
                        "dbo.fGetNearbyObjEq({},{},{}) as r on s.bestObjID = r.objID"\
                        .format(ra, dec, search_radius)
            spec_result = sqlcl.query(query_str).read()[8:]
            spec_data = np.genfromtxt(BytesIO(spec_result), names=True, delimiter=",", 
                                         dtype = query_dtypes)
            spec_data = np.atleast_1d(spec_data)
  
            # Mask out all objects from the quereied datasets that do not satisfy class == GALAXY
            print('spec gals: {}'.format(len(spec_data)))
            if(len(spec_data)!=0):
                galMask = spec_data['class'] == b'GALAXY'
                spec_data = spec_data[galMask]
            print('spec gals: {}'.format(len(spec_data)))
            print('z_noqso: {}'.format(sum(spec_data['z_noqso'])))
            
            np.save('{}{}_galaxies.npy'.format(saveDest, name), spec_data)       
            spec_count += len(spec_data)

        except ValueError as ie:
            fails.append(i)
            print("Failure on cluster {}".format(i))
    np.save('{}fails_indices.npy'.format(saveDest), fails)
    
    end = time.time()
    print("\nFinished. Query time: {:.2f} s\n{} total spec gals"
           .format(float(end - begin), spec_count))
