import pdb
import glob
import astropy
import numpy as np
import clstrTools as ct
import matplotlib as mpl
import interlopers as lop
from astropy.io import fits
import dispersionStats as stats
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as recfun
from astropy.cosmology import Planck15 as cosmo
from interlopers import NFW_escape_interlopers as nfwlopers 
sptPath = '/home/joe/skydrive/Work/HEP/data/spt'
sdssPath = '/home/joe/skydrive/Work/HEP/data/sdss'
clusterPath = '{}/sptgmos_clusters'.format(sptPath)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['text.usetex'] = True

def makeDiagram(maxR = 3, maxV = 4, norm = True):
    
    data  = np.load('{}/phaseSpace_SPT.npy'.format(sptPath))
    #data  = np.load('{}/phaseSpace_SDSS.npy'.format(sdssPath))
    #data = recfun.stack_arrays((data_spt,data_sdss), autoconvert=True, usemask=False) 
    r = data['r']
    v =data['v']
    types = data['type']
 
    # find indices of each galaxy type in data (boolean masks)
    bounds = np.logical_and([ri<=maxR for ri in r], [vi<=maxV for vi in v])
    r = r[bounds]
    v = v[bounds]
    types = types[bounds]
    passive = np.array([t == 'k_' for t in types])
    starForming = np.array(['e' in t for t in types])
    postStarBurst = np.array(['a' in t for t in types])
    masks = [passive, starForming, postStarBurst]
    dispAll = stats.bDispersion(v)
 
    numPass = len(passive[passive])
    numSF = len(starForming[starForming])
    numPSB = len(postStarBurst[postStarBurst])
    pops = [numPass, numSF, numPSB]
    colors = ['Reds', 'PuBu', 'BuGn']
    labels = ['Passive', 'StarForming', 'PostStarburst']
    bins = [22, 15, 15] 

    fig = plt.figure(1)

    for i in range(len(masks)):
    
        # gnenerate 2d histogram data
        plt.figure(1)
        ax = fig.add_subplot(3,1,i+1)
        plt.hold(True)
        radBins = np.linspace(0, maxR, bins[i]+1)
        velBins = np.linspace(0, maxV, bins[i]+1)
        hist = np.histogram2d(r[masks[i]], abs(v)[masks[i]], bins=[radBins, velBins], normed=False)
        pop = len(r[masks[i]])
        counts = hist[0].T

        # get velocity dispersion info
        dispPop = stats.bDispersion(v[masks[i]])
        dispRatio = dispPop / dispAll
 
        # normalize each radial bin by annulus area
        binAnnuli = [np.pi * (radBins[j+1]**2 - radBins[j]**2) for j in range(len(radBins)-1)]
        binWeights = 1/np.array(binAnnuli)
        countsNorm = (counts * binWeights)
        #pdb.set_trace()

        # get bin center positions
        dx = np.diff(radBins)[0]/2
        dy = np.diff(velBins)[0]/2
        binx = (radBins + dx)[:-1]
        biny = (velBins + dy)[:-1]
        Xcen,Ycen = np.meshgrid(binx, biny)
        Xedge,Yedge = np.meshgrid(radBins, velBins)

        # normalize histogram volume to unity to make this pdf
        if(norm):
            binArea = dx*dy*4
            volume = pop * binArea
            pdb.set_trace()
            counts = counts / volume

        # draw histogram
        histPlot = ax.pcolormesh(Xedge, Yedge, counts, cmap=colors[i])
        fig2 =plt.figure(2)
        ax2 = fig2.add_subplot(3,1,i+1)
        histPlot2 = ax2.pcolormesh(Xedge, Yedge, countsNorm, cmap=colors[i])
        
        # format plots
        fig.colorbar(histPlot)
        ax.text(0.7, 0.65, r'$\mathrm{{{0}}}$\\$\mathrm{{ {1}\>\>gals }}$\\' \
                            '$\sigma_{{ \mathrm{{v}} }} / \sigma_{{ \mathrm{{v,all}} }} = {2:.2f}$'
                .format(labels[i], pops[i], dispRatio), transform=ax.transAxes, fontsize=18)
        ax.set_xlabel(r'$r_{\mathrm{proj}} / R_{200}$', fontsize=18)
        ax.set_ylabel(r'$|v_{\mathrm{LOS}}| / \sigma_v$', fontsize=18)
        ax.set_xlim([0, maxR])
        ax.set_ylim([0, maxV])
        
        ax2.set_xlim([0, maxR])
        fig2.colorbar(histPlot2)
        ax2.set_ylim([0, maxV])

        # make contour plot
        #if(i < 2): ax2.contour(X, Y, counts, cmap = colors[i], linewidth=3)
        #ax2.set_xlabel(r'$r_{\mathrm{proj}} / R_{200}$', fontsize=18)
        #ax2.set_ylabel(r'$|v_{\mathrm{LOS}}| / \sigma_v$', fontsize=18)
        #ax2.set_xlim([0.1, 1.5])
        #ax2.set_ylim([0.1, 2])

    plt.show()


def LOSVelocity(gal_zs, host_z):
    '''
    Find LOS velocity of galaxy in km/s with respect to bulk cluster redshift
    
    :param z: array-like of galaxy redshifts
    :param host: host cluster redshift
    :return: numpy array of LOS velocities
    '''
   
    vLos = abs(ct.LOS_properVelocity(gal_zs, host_z))
    return vLos


def projDist(gals, host, host_z):
    '''
    Find projected distance of each galaxy to cluster center
    
    :param gals: array-like of galaxy properties, at least including coordinates 'RA' and 'Dec'
    :param host: array-like of host cluster properties, at least including 
                 coordinates 'RA' and 'Dec'
    :param host_z: host cluster redshift
    ''' 
    try:
        coords = [(gal['RA'], gal['Dec']) for gal in gals]
    except ValueError:
        # accomodate for slightly different column keywords in sdss data
        coords = [(gal['ra'], gal['dec']) for gal in gals]
    center = (host['RA'], host['Dec'])
    rProj = ct.projectedDist(coords, center, host_z)
    return rProj


def phaseSpace():
    
    print('Processing SPT data')
    phaseSpaceSPT()
    #print('Processing ACT data')
    #phaseSpaceACT()
    #print('Processing SDSS data')
    #phaseSpaceSDSS()
    makeDiagram()

def phaseSpaceSPT():
    
    #---------- SPT ----------
    
    # Gather data
    clusters = sorted(glob.glob('{}/SPT*prop*'.format(clusterPath)))
    clusterGals = sorted(glob.glob('{}/SPT*gal*'.format(clusterPath)))
    all_vLos_norm = []
    all_rProj_norm = []
    all_types = []

    # loop through each cluster 
    for n in range(len(clusters)):

        gals = np.load(clusterGals[n])
        host = np.load(clusters[n])
        if(len(host.dtype)) < 5: continue
        print('Working on SPT cluster {} ({}/{})'.format(host['spt_id'], n, len(clusterGals)))

        #vLos = LOSVelocity(gals['z'], host['sz_z'])
        vLos = gals['v']
        if(len(gals) < 10):
            rProj = projDist(gals, host, host['sz_z'])
        else:
            rProj = projDist(gals, host, host['spec_z'])
        vLos_norm = vLos / host['v_disp'] 
        rProj_norm = rProj / host['r200']
     
        # -------------------- clip interlopers --------------------
        #start_pop = len(vLos)
        #[mask, escFunc] = nfwlopers(rProj, vLos, host['r200'][0], host['m200'][0], host['sz_z'][0]) 
        #end_pop = len(vLos[mask])
        
        #members = gals[mask]
        #interlopers = gals[~mask]
        #all_vLos_norm += list(vLos_norm[mask])
        #all_rProj_norm += list(rProj_norm[mask]) 
        #all_types += list(gals['gal_type'][mask]) 
        
        #print('removed {}/{} galaxies as interlopers'.format(start_pop-end_pop, start_pop))
        
        mask = gals['mem?']
        all_vLos_norm += list(vLos_norm[mask])
        all_rProj_norm += list(rProj_norm[mask])
        all_types += list(gals['gal_type'][mask])
    
    # save data
    cols = ['v', 'r', 'type']
    all_vLos_norm = np.array(all_vLos_norm)
    all_rProj_norm = np.array(all_rProj_norm)
    all_types = np.array(all_types)
    data = np.rec.fromarrays([all_vLos_norm, all_rProj_norm, all_types], names=cols)
    np.save('{}/phaseSpace_SPT.npy'.format(sptPath), np.array(data))
    print('Done')


def phaseSpaceACT():
    pass


def phaseSpaceSDSS():
    
    #---------- SDSS ----------
    
    # Gather data
    galsPath = '../../data/sdss/sdss_spec_catalog'
    redMapper = '../../data/sdss/redMapper_catalog.fits'

    galsFiles = sorted(glob.glob('{}/*galaxies.npy'.format(galsPath)))
    ids = [g.split('_')[-2].split('/')[-1] for g in galsFiles]
        
    all_clusters = fits.open(redMapper)[1].data
    cluster_indices = np.array([np.where(all_clusters['NAME'] == i)[0][0] for i in ids])
    clusters = all_clusters[cluster_indices]
    all_vLos_norm = []
    all_rProj_norm = []
    all_types = []

    for n in range(len(clusters)):
        
        host = clusters[n]
        gals = np.load(galsFiles[n])
        if(len(gals)  < 5): continue
        print('Working on SDSS cluster {} ({}/{})'.format(host['NAME'], n, len(clusters)))
  
        start_pop = len(gals)
        [v, host_z, mask] = lop.sig_interlopers(gals['z'])
        end_pop = len(v)

        if(len(gals) >= 10):
            host_vdisp = stats.bDispersion(v)
            print('>10 gals, using z = {} (z_photo = {})'.format(host_z, host['Z_LAMBDA']))
        else:
            host_vdisp = stats.gDispersion(v)
            host_z = host['Z_LAMBDA'] 

        host_m200 = ct.richness_to_m200(host['LAMBDA'])
        host_r200 = ct.mass_to_radius(host_m200, host_z, h=70, mdef='200c')
        vLos = LOSVelocity(gals['z'], host_z)
        rProj = projDist(gals, host, host_z)
        vLos_norm = vLos / host_vdisp 
        rProj_norm = rProj / host_r200
        
        all_vLos_norm += list(vLos_norm[mask])
        all_rProj_norm += list(rProj_norm[mask]) 
        all_types += list(gals['gal_type'][mask]) 
        print('removed {}/{} galaxies as interlopers'.format(start_pop-end_pop, start_pop))
        
    # save data
    cols = ['v', 'r', 'type']
    all_vLos_norm = np.array(all_vLos_norm)
    all_rProj_norm = np.array(all_rProj_norm)
    all_types = np.array(all_types)
    data = np.rec.fromarrays([all_vLos_norm, all_rProj_norm, all_types], names=cols)
    np.save('{}/phaseSpace_SDSS.npy'.format(sdssPath), np.array(data))
    print('Done')
