'''
Joe Hollowed
COSMO-HEP 2017

Here are a collection of validation test functions for the protoDC2 galaxy catalog and
cluster catalog. This includes statistical tests to ensure correct physics output 
(i.e, realistic velocity distribution information), and also bug checks which represent issues 
which we have run into during development. 
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
rcParams.update(params)

import pdb
import h5py
import time
import numpy as np
import pylab as plt
import simTools as st
import clstrTools as ct
import dispersionStats as stat
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.mplot3d import Axes3D
from astropy.cosmology import WMAP7 as cosmo
from matplotlib.ticker import ScalarFormatter


# ===================================================================================================
# ============================== velocity segregation by galaxy color ===============================
# ===================================================================================================


def color_segregation_test(mask_var = 'gr', mask_magnitude = False, include_error = True):
    '''
    Use the protoDC2 stacked cluster as output from make_protoDC2_stackedCluster to 
    segregate galaxy population by color (cut in g-r rest magnitue space). Pass the 
    resultant color-dependent information to color_segregation_plot() to compare 
    the dispersion statistics of each population (red/blue).
    
    :param mask_var: what galaxy property to preform the color cut on. Default is 'gr', 
                     meaning that the cut will be preformed in g-r space. Other valid 
                     options are 'sfr' (which will cut at the median of this value)
    :param mask_magnitude: whether or not to preform a magnitude cut at -16 r-band
    :param include_error: whether or not to run a bootstrap routine to get confidence intervals 
                          on dispersions
    :return: None
    '''

    valid_maskVars = ['gr', 'sfr']
    if(mask_var not in valid_maskVars):
        raise ValueError('mask_var of \'{}\' is not accepted. Available masking properties'\
                         ' are {}'.format(mask_var, valid_maskVars))
    
    protoDC2_stack = h5py.File('data/protoDC2_haloStack_shear_nocut_A.hdf5', 'r')
    vel = protoDC2_stack['pecV_normed'][:]
    dist = protoDC2_stack['projDist_normed'][:]

    # do galaxy cut
    if(mask_var == 'gr'):
        g_band = protoDC2_stack['magnitude:SDSS_g:rest'][:]
        r_band = protoDC2_stack['magnitude:SDSS_r:rest'][:]
        gr_color = g_band - r_band
        colorMask = gr_color > 0.25
    elif(mask_var == 'sfr'):
        sfr = protoDC2_stack['totalStarFormationRate'][:]
        colorMask = sfr < np.median(sfr)
    if(mask_magnitude):
        magMask = rband > -16
        colorMask = colorMask & magMask

    vRed = vel[colorMask]
    vBlue = vel[~colorMask]
    dRed = dist[colorMask]
    dBlue = dist[~colorMask]

    # measure dispersions and error. I use 'o' to mean "dispersion" since
    # it's breif and kinda looks like a sigma
    if(include_error):
        tot_o, tot_o_err = stat.bootstrap_bDispersion(vel)
        red_o, red_o_err = stat.bootstrap_bDispersion(vRed)
        blue_o, blue_o_err = stat.bootstrap_bDispersion(vBlue)
    else:
        tot_o = stat.bDispersion(vel)
        red_o = stat.bDispersion(vRed)
        blue_o = stat.bDispersion(vBlue)

    # take the ratio of the red and blue dispersions with that of the entire population, 
    # and find the random error in said ratios by standard error propegation
    redRatio = red_o / tot_o
    blueRatio = blue_o / tot_o
    if(include_error):
        redRatio_err = redRatio * np.sqrt( (red_o_err/red_o)**2 + (tot_o_err/tot_o)**2 )
        blueRatio_err = blueRatio * np.sqrt( (blue_o_err/blue_o)**2 + (tot_o_err/tot_o)**2 )
    else:
        redRatio_err = None
        blueRatio_err = None

    #np.savez('{}_results.npz'.format(mask_var), vRed=vRed, vBlue=vBlue, dRed=dRed, dBlue=dBlue, 
    #         redRatio=redRatio, blueRatio=blueRatio, mask_var=mask_var, redRatio_err=redRatio_err, 
    #         blueRatio_err=blueRatio_err)
   
    # plot segregation results as velocity distributions and phase space diagrams
    color_segregation_plot(vRed, vBlue, redRatio, blueRatio, mask_var, redRatio_err, blueRatio_err)
    phaseSpace_segregation_plot(vRed, vBlue, dRed, dBlue, mask_var)


# -------------------------------------------------------------------------------------------------


def color_segregation_plot(vr, vb, vDispr, vDispb, var, vDispr_err=None, vDispb_err=None):
    '''
    Plot the velocity normed distributions for both red and blue galaxy populations
    from the protoDC2 stacked cluster. This function is meant to be called by color_segregation_test()

    :param vr: LOS peculiar velocities of red galaxies normalized by their host-halo velocity-dispersion
    :param vb: same as vr for blue galaxies
    :param vDispr: the velocity-dispersion of the red population over the velocity-dispersion 
                   of the total population
    :param vDispb: same as vDispr for blue galaxies
    :param var: the variable used to determine red/blue galaxy cut
    :param vDispr_err: the random error in vDispr
    :param vDispb_err: the random error in vDispb
    :return: None
    '''

    if(vDispr_err ==None and vDispb_err == None):
        ansRed = '{:.2f}'.format(vDispr)
        ansBlue = '{:.2f}'.format(vDispb)
    else:
        ansRed = r'{:.2f} \pm {:.3f}'.format(vDispr, vDispr_err)
        ansBlue = r'{:.2f} \pm {:.3f}'.format(vDispb, vDispb_err)

    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    
    ax.hist(vr, 100, histtype='step', color='r', normed=True)
    ax.hist(vb, 100, histtype='step', color='b', normed=True)
    ax.text(-7.5, 0.26, r'$\sigma_{{ v,\mathrm{{red}} }} / \sigma_{{ v,\mathrm{{all}} }}'\
                       '= {}$'.format(ansRed), fontsize=18)
    ax.text(-7.5, 0.24, r'$\sigma_{{ v,\mathrm{{blue}} }} / \sigma_{{ v,\mathrm{{all}} }}'\
                       '= {}$'.format(ansBlue), fontsize=18)
    ax.set_xlabel(r'$v/\sigma_{v,\mathrm{all}}$', fontsize=22)
    ax.set_ylabel('pdf', fontsize=18)
    plt.grid()
    #plt.show()


# -------------------------------------------------------------------------------------------------


def phaseSpace_segregation_plot(vr, vb, dr, db, var):
    '''
    Plot all galaxies in radial distance-LOS velocity phase space, with two
    colored populations corresponding to red/blue galaxies
    
    :param vr: LOS peculiar velocities of red galaxies normalized by their host-halo velocity-dispersion
    :param vb: same as vr for blue galaxies
    :param dr: projected radial distance of red galaxies nomalized by their host-halo r200 distance
    :param db: same as dr for blue galaxies
    :param var: the variable used to determine red/blue galaxy cut
    :return: None
    '''
    
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    
    bins = np.linspace(0, 2, 30)
    ax.hist2d(dr, vr, bins=bins, normed=True, cmap='Reds', alpha=0.5)
    ax.hist2d(db, vb, bins=bins, normed=True, cmap='Blues', alpha=0.5)
    plt.show()


# ===================================================================================================
# =============================== mass - velocity-dispersion relation ===============================
# ===================================================================================================


def mass_vDisp_relation(mask_magnitude = False):
    '''
    This function was written as a validation test for the protoDC2 catalog - it uses 
    the observed galaxy redshifts that I have created in the catalog to measure galaxy-based
    velocity-dispersions for cluster-sized halos, and compares the results to the particle-based
    SOD velocity-dispersions. If the galaxies are placed correctly, and the observed redshifts are 
    constructed correctly, then we should see the vDisp-Mass relation roughly match, whether we are 
    using galaxy or particle based dispersions. This rough match should include a systematic bias in
    the galaxy-based dispersions. This function is meant to call makePlots() above.

    :param mask_magnitude: whether or not to preform a magnitude cut at -16 r-band
    :return: None
    '''
    
    protoDC2 = h5py.File('data/protoDC2_clusters_shear_nocut_A.hdf5', 'r')
    halos = list(protoDC2.keys())
    n = len(halos)
    galDisp = np.zeros(n)
    sodDisp = np.zeros(n)
    realMass = np.zeros(n)
    ez = np.zeros(n)
    
    for j in range(n):
        halo = protoDC2[halos[j]]

        # get halo properties, including particle-based dispersion and mass
        zHost = halo.attrs['halo_z']
        zHost_err = halo.attrs['halo_z_err']
        aHost = 1/(1+zHost)
        ez[j] = cosmo.efunc(zHost)
        realMass[j] = halo.attrs['sod_halo_mass'] * ez[j]
        sodDisp[j] = halo.attrs['sod_halo_vel_disp'] * aHost
        
        # mask faint galaxies (less negative than -16 in rest r-band)
        rMag = halo['magnitude:SDSS_r:rest'][:]
        if(mask_magnitude): rMag_mask = (rMag > -16)
        else: rMag_mask = np.ones(len(rMag), dtype=bool)

        # use redshifts of galaxies surviving the mask to calculate galaxy-based dispersion
        z = halo['redshiftObserver'][:][rMag_mask]
        pecV = ct.LOS_properVelocity(z, zHost)
        galDisp[j]  = stat.bDispersion(pecV) * aHost
 
    plot_mass_vDisp_relation(galDisp, sodDisp, realMass)
    plot_vDisp_comparison(galDisp, sodDisp)


# -------------------------------------------------------------------------------------------------


def plot_mass_vDisp_relation(sigma, sigmadm, realMass):
    '''
    Plot the mass-vDisp relation for each halo mass passed, using both the galaxy and particle-based
    velocity dispersions. This function is meant to be called by mass_vDisp_relation() above.

    :param sigma: An array of galaxy-based dispersions
    :param sigmadm: An array of particle-based dispersions
    :param realMass: the SO mass of the halo
    '''

    
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(realMass, sigmadm, 'xm', ms=6, mew=1.6, alpha=0.5, label='particle-based dispersions')
    ax.plot(realMass, sigma, '.g', ms=6, alpha=0.5, label='galaxy-based dispersions')

    fitMass = np.linspace(3e13, 8e14, 100)
    fit = lambda sigDM15,alpha: sigDM15 * ((fitMass) / 1e15)**alpha
    fitDisp = fit(1082.9, 0.3361)

    ax.plot(fitMass, fitDisp, '--k', lw=2, label='Evrard+ 2003')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([200, 1300])
    ax.set_xlim([3e13, 8e14])
    ax.grid()

    ax.yaxis.set_major_formatter(ScalarFormatter())
    locy = plticker.MultipleLocator(base=100)
    ax.yaxis.set_major_locator(locy)
    locx = plticker.MultipleLocator(base=5e13)
    ax.xaxis.set_major_locator(locx)

    ax.set_ylabel(r'$\sigma_v$ (km/s)', fontsize=20)
    ax.set_xlabel(r'$m_{200}$ (M$_{sun} h^{-1}$)', fontsize=20)
    ax.legend(loc='upper left', fontsize=12)
    plt.show()


# -------------------------------------------------------------------------------------------------


def plot_vDisp_comparison(sigma, sigmadm):
    '''
    Plot the particle_based vs glaxy-based velocity distribution for each halo as passed.
    This function is meant to be called by mass_vDisp_relation() above.

    :param sigma: An array of galaxy-based dispersions
    :param sigmadm: An array of particle-based dispersions
    '''
    
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(sigmadm, sigma, 'xb', ms=6, mew=1.6, alpha=0.5)
    ax.plot([200, 1300], [200, 1300], '--k', lw=2)

    ax.set_ylim([200, 1000])
    ax.set_xlim([200, 1000])
    ax.grid()

    ax.set_ylabel(r'$\sigma_{v,\mathrm{particles}}$ (km/s)', fontsize=20)
    ax.set_xlabel(r'$\sigma_{v,\mathrm{galaxies}}$ (km/s)', fontsize=20)
    plt.show()


# ===================================================================================================
# =============================== check peculiar redshifts on cores =================================
# ===================================================================================================


def test_pecZ_cores():
    '''
    This function checks the calculation preformed in the script calc_peculiar_z.py
    using the core-catalog (no light-cone). Cores are plotted in 3D comoving space 
    (x, y, z) as points, and a velocity vector is drawn from each core. The color 
    of the velocity vector represents whether the core was found to be blue- or 
    red-shifted in peculiar velocity. A larger balck vector points in the direction 
    of the observer.
    '''
        
    corePath = '/home/jphollowed/code/repos/cosmology/core_tracking/data/coreCatalog/'\
               'MedianVelocity/haloCores_processed.hdf5'
    f = h5py.File(corePath, 'r')
    step = 293
    zTool = st.StepZ(200, 0, 500)
    nHalos = len(list(f['step_{}'.format(step)].keys()))

    for j in range(nHalos):
     
        halo = f['step_{}'.format(step)][list(f['step_{}'.format(step)].keys())[j]]
        center = np.array([halo.attrs['fof_halo_center_{}'.format(r)] for r in ['x', 'y', 'z']])
        x = halo['x'][:]
        y = halo['y'][:]
        z = halo['z'][:]
        vx = halo['vx'][:]
        vy = halo['vy'][:]
        vz = halo['vz'][:]

        rs = np.ones(len(vx))
        rs = rs * zTool.get_z(step)
        if(len(rs) < 50): continue
        z_pec, z_tot, v_pec, v_los, r_rel_mag, r_rel_prop, r_dist = pecZ.pecZ(x, y, z, vx, vy, vz, rs)

        fig = plt.figure(1)
        fig.clf()
        ax = fig.gca(projection='3d')
        color = [['r', 'b'][rs < 0] for rs in z_pec]

        for i in range(len(z_pec)):
                ax.quiver(x[i], y[i], z[i], vx[i], vy[i], vz[i], color=color[i], length = v_pec[i]/1000,
                          arrow_length_ratio = 0.1, pivot='tail', lw=1.3)
        ax.plot(x, y, 'o', zs=z, color='grey', mew=0)

        mx = np.median(x)
        my = np.median(y)
        mz = np.median(z)
        ax.quiver(mx, my, mz, 0-mx, 0-my, 0-mz, color='k', length=2.2, arrow_length_ratio = 0.14, 
                  pivot='tail', lw=2.5)
        ax.set_xlim([np.median(x)-1.5, np.median(x)+1.5])
        ax.set_ylim([np.median(y)-1.5, np.median(y)+1.5])
        ax.set_zlim([np.median(z)-1.5, np.median(z)+1.5])

        fig.clf()
        ax1 = fig.add_subplot(111)
        bins = np.histogram(r_dist, 20)[1]
        ax1.hist(r_rel_prop, bins=bins, color=[1, .2, .2], linestyle='--', histtype='step')
        ax1.hist(r_dist, bins=bins, color='red', histtype='step')
        plt.show()


# ===================================================================================================
# =============================== check for many centrals bug========================================
# ===================================================================================================


def manyCentrals_bug():
    '''
    This function uses the protoDC2 cluster catalog to check for the presence 
    of halos in which there are many galaxies marked as "centrals" (there should
    be no more or less than 1)
    '''

    f = h5py.File('data/protoDC2_catalog.hdf5', 'r')
    halos = list(f.keys())
    numCen = np.zeros(len(halos))
    numMatch = np.zeros(len(halos))
    rat = np.zeros(len(halos))
    steps = np.zeros(len(halos))
    zs = np.zeros(len(halos))

    for i in range(len(halos)):
        halo = f[halos[i]]
        centrals = halo['nodeIsIsolated'][:]
        totCentrals = sum(centrals)
        numCen[i] = totCentrals

        galIdx = halo['nodeIndex'][:]
        hostIdx = halo.attrs['hostIndex']
        steps[i] = halo.attrs['step']
        zs[i] = halo.attrs['z']
        totIdxMatch = sum(galIdx == hostIdx)
        numMatch[i] = totIdxMatch
        
        uniqueIdx = np.unique(galIdx)
        rat[i] = len(uniqueIdx) / len(galIdx)   
        

    badMask = numCen == 1 
    pdb.set_trace()
