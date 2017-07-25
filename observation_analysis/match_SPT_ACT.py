# Joe Hollowed
# HEP 6/2017

import pdb
import numpy as np
import astropy.io.fits as fits
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

def matchClusters():
    '''
    Plot SPT clusters in RA-Dec space and draw 3-arcmin regions around each one
    Plot ACT clusters on the same set of axes - any clusters that fall within SPT 3-arcmin 
    regions are considered to be the same cluster.
    '''

    # load data
    dataPath = '/home/joe/skydrive/Work/HEP/data/spt'
    sptFits = '{}/SPT-SZ.fits'.format(dataPath)
    sptgmosFits = '{}/sptgmos_clusters_unmatched.fits'.format(dataPath)
    spt = fits.open(sptFits)[1].data
    sptgmos = fits.open(sptgmosFits)

    # get cluster ids
    gmosIds = np.array([sptgmos[i+1].header['EXTNAME'] for i in range(len(sptgmos)-1)])
    actMask = np.array(['ACT' in gid for gid in gmosIds])
    actIds = gmosIds[actMask]
    print('Found {} ACT clusters'.format(len(actIds))) 
   
    # define RA, and Dec for ACT clusters
    actRA = [94.1500, 46.0625, 39.2625, 52.7250, 56.7125, 33.8250, 38.1875, 106.8042, 38.9667]
    actDec = [-52.4597, -49.3617, -49.6575, -52.4678, -54.6483, -52.2083, -52.9522, -55.3800, -51.3544]

    # get spt clusters in vicinity of ACT clusters
    sptRAMask = np.logical_and([ra > min(actRA) - 1 for ra in spt['RA']], 
                             [ra < max(actRA) + 1 for ra in spt['RA']])
    sptDecMask = np.logical_and([dec > min(actDec) - 1 for dec in spt['Dec']], 
                             [dec < max(actDec) + 1 for dec in spt['Dec']])
    sptMask = np.logical_and(sptRAMask, sptDecMask)

    spt = spt[sptMask]
    sptRA = spt['RA']
    sptDec = spt['DEC']
    sptIds = spt['SPT_ID']

    # find angular separation of each ACT cluster to each SPT cluster in arcmin
    actCoords = SkyCoord(ra=actRA*u.degree, dec=actDec*u.degree)
    sptCoords = SkyCoord(ra=sptRA*u.degree, dec=sptDec*u.degree)
    separation = np.array([actCoords.separation(sptCoords[i]).value * 60 for i in range(len(sptIds))])
    
    # find matches (centers within 3 arcmin)
    matchingIds = np.array([sptIds[separation.T[i]<3] for i in range(len(actIds))])
    matchMask = np.array([len(match)!=0 for match in matchingIds])
    matchingClusters = np.array(sptgmos[1:])[actMask][matchMask]
    print('Found {} ACT clusters with SPT matched\n'.format(len(matchingClusters)))
    pdb.set_trace() 
    # rename matched clusters
    for j in range(len(matchingClusters)):
        cluster = matchingClusters[j]
        old_name = cluster.header['EXTNAME']
        cluster.header['EXTNAME'] = matchingIds[matchMask][j][0]
        print('Changed {} to {}'.format(old_name, cluster.header['EXTNAME']))
    
    newGals = sum([len(s.data) for s in np.array(sptgmos[1:])[actMask][matchMask]])
    print('\n{} new gals belonging to SPT clusters'.format(newGals))

    # save renamed clusters to new FITS
    np.array(sptgmos[1:])[actMask][matchMask] = matchingClusters 
    sptgmos.writeto('{}/sptgmos_clusters.fits'.format(dataPath), clobber=True)
    print('Done')
