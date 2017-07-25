
import numpy as np
import sqlcl
import pyfits
from io import BytesIO
import time
import os
import clstrTools as ct
import pdb

def querySDSS(mask_bcgs = 1, mask_richest = 0, richest_num = 100, start = 0, num = 0, r200_factor = 1, bcg_arcsec = 5):
    '''
    Query the SDSS data release for spectral objects in redmapper identified clusters
    :param mask_bcgs: boolean; whether or not to only query clusters than have a BCG redshift
    :param mask_richest: boolean; whether or not to only query n richest clusters
    :param richest_num: int; how many clusters to use in the richness mask
    :param start: Cluster index to begin query at (if querying different clusters in different runs,
                  may have unexpected behavior if mask parameters are different between those runs)
    :param num: number of clusters to query, starting at 'start'
    :param r200_factor: multiples of r200 to use as query search radius
    :param bcg_arcsec: how many arcseconds to search around center of cluster (bcg coordinates)
    :return: None. Save query results to numpy files.
    '''

    print('Reading fits')
    catFits = pyfits.open("../data/sdss/redMapper_catalog.fits")
    if(mask_bcgs == 0): saveDest = '../output/sdssCatalog_bcgs/'
    elif (mask_bcgs == 1): saveDest = '../output/sdssCatalog_nobcgs/'
    else: saveDest = '../output/sdssCatalog/'
    #header list can be found in http://arxiv.org/pdf/1303.3562v2.pdf

    #data from fits
    data =  catFits[1].data

    #collect cluster richness and most likely bcg spec z's
    cat_lambda = data.field('lambda')
    bcg_z = data.field('z_spec')

    #find clusters that have known likley-bcg redshifts (z_spec != -1)
    #find 100 richest clusters
    print('Building masks')

    #get all clusters with and without a bcg redshift
    if(mask_bcgs == 1): bcg_mask = (bcg_z != -1)
    elif(mask_bcgs == -1): bcg_mask = (bcg_z == -1)
    else: bcg_mask = np.ones(len(cat_lambda), dtype=bool)

    #get all clusters that are in the richest_num richest clusters (highest lambda values)
    if(mask_richest == 1):
        richestIdx = np.argsort(-cat_lambda)[0:richest_num]
        allIdx = np.r_[0:len(cat_lambda)]
        rich_mask = np.array([r in richestIdx for r in allIdx], dtype=bool)
    else:
        rich_mask = np.ones(len(cat_lambda), dtype=bool)

    mask = np.bitwise_and(bcg_mask, rich_mask)


    #collect cluster data and mask it
    print('Collecting Cluster data')
    cat_lambda = cat_lambda[mask]
    cat_lambdaErr = data.field('lambda_err')[mask]
    cat_ids = data.field('name')[mask]
    cat_ra = data.field('ra')[mask]
    cat_dec = data.field('dec')[mask]
    cat_z = data.field('z_lambda')[mask]
    cat_zErr = data.field('z_lambda_err')[mask]
    bcg_z = bcg_z[mask]
    #following bcg properties are all given for the top 5 most likely centers (5 x float)
    bcg_pcen = data.field('p_cen')[mask]
    bcg_ra = data.field('ra_cen')[mask]
    bcg_dec = data.field('dec_cen')[mask]
    bcg_ids = data.field('id_cen')[mask]

    fails = []
    if num == 0: num = len(cat_lambda)
    if num > len(cat_lambda): num = len(cat_lambda)
    query_table = "SpecObjAll"
    query_dtypes = ['int64', 'f8', 'f8', 'f4', 'f4', 'int', 'a8', 'a8', 'f4', 'f4', 'f4', 'f4', 'f4',
                    'int', 'a8', 'a8', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4']

    # query_dtypes = [('bestObjID', 'int64'),('ra', 'f8'),('dec', 'f8'),('z', 'f4'),('zErr', 'f4'),
    #                 ('zWarning', 'int'),('class', 'bytes'),('subClass', 'bytes'),('rChi2', 'f4'),
    #                 ('DOF', 'f4'),('rChi2Diff', 'f4'),('z_noqso', 'f4'),('zErr_noqso', 'f4'),
    #                 ('zWarning_noqso', 'int'),('class_noqso', 'bytes'),('subClass_noqso', 'bytes'),
    #                 ('rChi2Diff_noqso', 'f4'),('velDisp', 'f4'),('velDispErr', 'f4'),('velDispZ', 'f4'),
    #                 ('velDispZErr', 'f4'),('velDispChi2', 'f4')]

    # query for num clusters, starting at index start
    for i in range(start, num):
        try:
            start = time.time()

            # query columns are defined here:
            # http://cas.sdss.org/dr12/en/help/browser/browser.aspx#&&history=description+SpecObjAll+U
            name = cat_ids[i]
            ra = cat_ra[i]
            dec = cat_dec[i]
            z = cat_z[i]
            zErr = cat_zErr[i]
            richness = cat_lambda[i]
            richnessErr = cat_lambdaErr[i]
            rad = ct.richness_to_arcmin(richness, z)
            mass = ct.richness_to_m200(richness)
            r200 = ct.mass_to_radius(mass, z, h = 70, mdef = '200c')
            search_radius = r200_factor * rad

            # save target properties
            cat = np.array([(ra, dec, z, zErr, richness, richnessErr, mass, rad, r200, bcg_z[i])],
                           dtype=[('ra', 'f8'), ('dec', 'f8'), ('cl_photo_z', 'f4'), ('cl_photo_zErr', 'f4'), ('lambda', 'f4'),
                                  ('lambdaErr', 'f4'), ('mass', 'f4'), ('rad_arcmin', 'f4'), ('r200', 'f4'),
                                  ('bcg_z', 'f4')])
            np.save('{}{}_prop{}.npy'.format(saveDest, name, i), cat)

            cat_bcg = np.array([['ids', 'ra', 'dec', 'pcen'], bcg_ids[i], bcg_ra[i], bcg_dec[i], bcg_pcen[i]])
            np.save('{}{}_bcg_prop{}.npy'.format(saveDest, name, i), cat_bcg)

            # Query and save all cluster objects (r200_factor * r200 arcmin search radius)
            print("\nQuerying spec objects within {:.2f} arcminutes of cluster {} (of {})".format(search_radius, i + 1, num))
            query_str = "select p.bestObjID, p.ra, p.dec, p.z, p.zErr, p.zWarning, p.class, p.subClass, p.rChi2, p.DOF, p.rChi2Diff, " \
                        "p.z_noqso, p.zErr_noqso, p.zWarning_noqso, p.class_noqso, p.subClass_noqso, p.rChi2Diff_noqso, p.velDisp, p.velDispErr, " \
                        "p.velDispZ, p.velDispZErr, p.velDispChi2 from {} p join dbo.fGetNearbyObjEq({},{},{}) r " \
                        "on p.bestObjID = r.ObjID".format(query_table, ra, dec, search_radius)
            cluster_result = sqlcl.query(query_str).read()[8:]

            cluster_data = np.genfromtxt(BytesIO(cluster_result), names=True, delimiter=",", dtype = query_dtypes)

            np.save('{}{}_data{}.npy'.format(saveDest, name, i), cluster_data)

            if(bcg_z[i] != -1):
                # Query and save all central cluster objects (bcg_arcsec search radius) if current cluster has an
                # identified bcg. If bcg_mask is set to 1, this block should always execute
                print("Querying spec objects within {:.2f} arcseconds of cluster {} (of {})".format(bcg_arcsec, i + 1, num))
                bcg_query_str = "select p.bestObjID, p.ra, p.dec, p.z, p.zErr, p.zWarning, p.class, p.subClass, p.rChi2, p.DOF, p.rChi2Diff, " \
                            "p.z_noqso, p.zErr_noqso, p.zWarning_noqso, p.class_noqso, p.subClass_noqso, p.rChi2Diff_noqso, p.velDisp, p.velDispErr, " \
                            "p.velDispZ, p.velDispZErr, p.velDispChi2 from {} p join dbo.fGetNearbyObjEq({},{},{}) r " \
                            "on p.bestObjID = r.ObjID".format(query_table, ra, dec, bcg_arcsec/60)
                center_result = sqlcl.query(bcg_query_str).read()[8:]

                center_data = np.genfromtxt(BytesIO(center_result), names=True, delimiter=",", dtype = query_dtypes)
                np.save('{}{}_{}center_data{}.npy'.format(saveDest, name, bcg_arcsec, i), center_data)

            end = time.time()
            print("\nFinished. Query time: {:.2f} s".format(float(end - start)))

        except ValueError as ie:
            fails.append(i)
            print("Failure")
        np.save('{}fails_indices.npy'.format(saveDest), fails)

#start script (adjust parameters here)
if __name__ == '__main__':
    querySDSS(mask_bcgs=1)
