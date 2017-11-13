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
import plotTools as pt
import clstrTools as ct
import coreTools as cot
import matplotlib as mpl
import calc_peculiar_z as cpz
import dispersionStats as stat
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.mplot3d import Axes3D
from astropy.cosmology import WMAP7 as cosmo
from matplotlib.ticker import ScalarFormatter


# ===================================================================================================
# ============================== velocity distribution by galaxy color ==============================
# ===================================================================================================


def color_distributions_test(mask_var = 'gr', mask_magnitude = False, include_error = True):
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
        magMask = r_band > -16
    else:
        magMask = np.ones(len(r_band))

    vRed = vel[colorMask][magMask]
    vBlue = vel[~colorMask][magMask]
    dRed = dist[colorMask][magMask]
    dBlue = dist[~colorMask][magMask]

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
    color_distributions_plot(vRed, vBlue, redRatio, blueRatio, mask_var, redRatio_err, blueRatio_err)
    phaseSpace_colors_plot(vRed, vBlue, dRed, dBlue, mask_var)


# -------------------------------------------------------------------------------------------------


def color_distributions_plot(vr, vb, vDispr, vDispb, var, vDispr_err=None, vDispb_err=None):
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


def phaseSpace_color_plot(vr, vb, dr, db, var):
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
# ========================== velocity segregation and compare to spt=================================
# ===================================================================================================

def organize_velocity_data():
    '''
    Pilot function meant to run velocity_segregation_test for real and mock data, and then run 
    plotting routines
    '''

    sptStack = h5py.File('../observation_analysis/spt/data/sptgmos_stackedCluster.hdf5', 'r')
    protoDC2_stack = h5py.File('data/protoDC2_haloStack_shear_nocut_A.hdf5', 'r')
    v_pdc2 = protoDC2_stack['pecV_normed']
    interloperMask = sptStack['member'][:]
    v_spt = sptStack['v_norm'][interloperMask]

    pdc2_sfr = protoDC2_stack['totalStarFormationRate'][:]
    colorMask_pdc2 = pdc2_sfr < np.median(pdc2_sfr)
    colorMask_spt = sptStack['gal_type'][interloperMask] == b'k_'

    if(not os.path.isfile('data/pdc2_segr.npz')):
        print('measuring protoDC2 segregation')
        p_pdc2, v_pdc2, v_pdc2_err = velocity_segregation_test(v_pdc2, colorMask_pdc2)
        np.savez('data/pdc2_segr.npz', p_pdc2=p_pdc2, v_pdc2=v_pdc2, v_pdc2_err=v_pdc2_err)
    else:
        pdc2_file = np.load('data/pdc2_segr.npz')
        p_pdc2 = pdc2_file['p_pdc2']
        v_pdc2 = pdc2_file['v_pdc2']
        v_pdc2_err = pdc2_file['v_pdc2_err']
    if(not os.path.isfile('data/spt_segr.npz')):
        print('measuring spt segregation')
        p_spt, v_spt, v_spt_err = velocity_segregation_test(v_spt, colorMask_spt, obs=True)
        np.savez('data/spt_segr.npz', p_spt=p_spt, v_spt=v_spt, v_spt_err=v_spt_err)
    else:
        pdc2_file = np.load('data/spt_segr.npz')
        p_spt = pdc2_file['p_spt']
        v_spt = pdc2_file['v_spt']
        v_spt_err = pdc2_file['v_spt_err']
    velocity_segregation_plot(p_pdc2, p_spt, v_pdc2, v_spt, v_pdc2_err, v_spt_err)


def velocity_segregation_test(v, popMask, obs = False, resamples=14, conf = 68.3):
    '''
    This function preforms velocity segregation as presented in Bayliss+2016. In this 
    test, we begin with a population of galaxies whose velocity dispersion is VD. Next, 
    we split the population into red and blue subsamples. An ensemble distribution
    is then constructed from a% red galaxies, and b% blue galaxies (where a+b=100). The color ratio a/b 
    is varied from 0 to 1 in #resmaples steps, and at each step the velocity dispersion of the ensemble 
    distribution, VE, is measured. The ultimate return of this function is VE/VD as a function of a/b (two
    lists of length #resamples).
    In general, of course, this function can be executed to segregate on any galaxy property other than color, 
    such as magnitude, given the details of the parameter popMask.

    :param v: an array of galaxy LOS velocities
    :param popMask: a boolean mask indicating the color of each galaxy as in the array v. 
                     It is assumed that True = Red, and False = Blue
    :param conf: the confidence within which to return the error - default is 1sigma or 68.3%
    :return: See function docstring
    '''

    prefix = ['pdc2', 'spt'][obs]
    if( not os.path.isfile('data/{}_vDispAll.npz'.format(prefix))):
        print('file not found; computing dispersion for entire {} sample'.format(prefix))
        vDisp_all, vDisp_all_err = stat.bootstrap_bDispersion(v)
        np.savez('data/{}_vDispAll.npz'.format(prefix), vDisp_all=vDisp_all, vDisp_all_err=vDisp_all_err)
    else:
        print('vDispAll file found for {}'.format(prefix))
        vDispInfo = np.load('data/{}_vDispAll.npz'.format(prefix))
        vDisp_all = vDispInfo['vDisp_all']
        vDisp_all_err = vDispInfo['vDisp_all_err']

    pcen = np.linspace(0, 1, resamples)
    vrat = np.zeros(resamples)
    vrat_err = np.zeros(resamples)
    maxPop = int(min([np.sum(popMask), np.sum(~popMask)]))
    #resampleSize = int(min( max(maxPop/10, 200), maxPop ))
    resampleSize = int(maxPop/2)
    pop1_size = [int(f) for f in np.ceil(pcen * maxPop)]
    pop2_size = [int(f) for f in np.floor((1-pcen) * maxPop)]
  
    for j in range(len(pcen)):
        print('working on ensemble {}/{}'.format(j, len(pcen)))

        #pop1 = stat.random_choice_noreplace(v[popMask], 1000, pop1_size[j])
        #pop2 = stat.random_choice_noreplace(v[~popMask], 1000, pop2_size[j])
        #realizations = [np.hstack([pop1[i], pop2[i]]) for i in range(1000)]
        #if gaussian: vDisp_rlz = [np.std(realization) for realization in realizations]
        #else: vDisp_rlz = [stat.bDispersion(realization) for realization in realizations]
       
        pop1 = v[popMask][0:pop1_size[j]]
        pop2 = v[~popMask][0:pop2_size[j]]
        sample = np.hstack([pop1, pop2])
        vDisp = stat.bDispersion(sample)
        vrat[j] = vDisp / vDisp_all
        
        # estimate error in dispersion via bootstrap
        resamples = np.array([np.random.choice(sample, size=resampleSize, replace=False) for i in range(1000)])
        if(resampleSize > 15):
            vDisp_realizations = stat.bDispersion(resamples)
        else:
            vDisp_realizations = stat.gDispersion(resamples)

        
        #vDisp_err = np.std(vDisp_realizations)/np.sqrt(1000)
        scatter = vDisp_realizations - vDisp
        critPoints = [0+(100-conf)/2, 100-(100-conf)/2]
        critVals = [np.percentile(scatter, critPoints[i], interpolation='nearest') for i in range(2)]
        confidence = [vDisp - crit for crit in critVals]
        vDisp_err= np.mean(abs(confidence - vDisp)) 
        vrat_err[j] = vrat[j] * np.sqrt((vDisp_err/vDisp)**2 + (vDisp_all_err/vDisp_all)**2)
        print(vrat[j], vrat_err[j])
    
    return pcen, vrat, vrat_err
   

def velocity_segregation_plot(p_pdc2, p_spt, v_pdc2, v_spt, v_pdc2_err, v_spt_err, xlabel='\% \> Passive'):
    
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(p_spt, v_spt, marker='o', linestyle='-', color='m', ms=8, label='SPT-GMOS')
    ax.fill_between(p_spt, v_spt-v_spt_err, v_spt+v_spt_err, color='m', alpha=0.15)
    ax.plot(p_pdc2, v_pdc2, marker='s', linestyle='-', color='c', ms=8, label='ProtoDC2')
    ax.fill_between(p_pdc2, v_pdc2-v_pdc2_err, v_pdc2+v_pdc2_err, color='c', alpha=0.15)
    ax.legend()
    #ax.set_ylim([0.7, 1.3])
    ax.set_xlabel(r'$\mathrm{{ {} }}$'.format(xlabel), fontsize=24)
    ax.set_ylabel(r'$\sigma_v \> / \> \sigma_{v,\mathrm{All}}$', fontsize=24)
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
    the galaxy-based dispersions. This function is meant to call makePlots() below.

    :param mask_magnitude: whether or not to preform a magnitude cut at -16 r-band
    :return: None
    '''
    
    #protoDC2 = h5py.File('data/protoDC2_clusters_shear_nocut_A.hdf5', 'r')
    protoDC2 = h5py.File('/media/luna1/jphollowed/protoDC2/protoDC2_clusters_full_shear_nocut_dust_elg_shear2_mod.hdf5', 'r')
    halos = list(protoDC2['clusters'].keys())
    n = len(halos)
    galDisp = np.zeros(n)
    sodDisp = np.zeros(n)
    realMass = np.zeros(n)
    ez = np.zeros(n)
    
    for j in range(n):
        halo = protoDC2['clusters'][halos[j]]

        # get halo properties, including particle-based dispersion and mass
        zHost = halo.attrs['halo_z']
        zHost_err = halo.attrs['halo_z_err']
        aHost = 1/(1+zHost)
        ez[j] = cosmo.efunc(zHost)
        realMass[j] = halo.attrs['sod_halo_mass'] * ez[j]
        sodDisp[j] = halo.attrs['sod_halo_vel_disp'] * aHost
        
        # mask faint galaxies (less negative than -16 in rest r-band)
        #rMag = halo['magnitude:SDSS_r:rest'][:]
        #if(mask_magnitude): rMag_mask = (rMag > -16)
        #else: rMag_mask = np.ones(len(rMag), dtype=bool)
        rMag_mask = np.ones(len(halo['x'][:]), dtype=bool)

        # use redshifts of galaxies surviving the mask to calculate galaxy-based dispersion
        z = halo['redshift'][:][rMag_mask]
        pecV = ct.LOS_properVelocity(z, zHost)
        galDisp[j]  = stat.bDispersion(pecV) * aHost
 
    plot_mass_vDisp_relation(galDisp, sodDisp, realMass)


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
    #ax.plot(realMass, sigmadm, 'xm', ms=6, mew=1.6, alpha=0.5, label='particle-based dispersions')
    #ax.plot(realMass, sigma, '.g', ms=6, alpha=0.5, label='galaxy-based dispersions')
    ax.scatter(realMass/1e14, sigma, color='b', edgecolor=np.array([111,168,220])/255,
               alpha=0.6, s=30, marker='o', label='protoDC2')
    ax.set_xscale('log')
    ax.set_yscale('log')

    fitMass = np.linspace(2e13, 8e14, 100)
    fit = lambda sigDM15,alpha: sigDM15 * ((fitMass) / 1e15)**alpha
    fitErr = lambda sigDM15, alpha, sigDM15Err, alphaErr: \
        np.sqrt( ((fitMass/1e15)**alpha)**2 * sigDM15Err**2 + \
                 (sigDM15*(fitMass/1e15)**alpha*np.log(alpha))**2 * alphaErr**2)
    fitDisp = fit(1082.9, 0.3361)
    fitDispErr = fitErr(1082.9, 0.3361, 4, 0.0026)

    ax.plot(fitMass/1e14, fitDisp, '--k', lw=2, label='Evrard+ 2008')
    ax.fill_between(fitMass/1e14, fitDisp-fitDispErr, fitDisp+fitDispErr, color='k', alpha=0.2)
    ax.grid()
    
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    locy = plticker.MultipleLocator(base=100)
    ax.yaxis.set_major_locator(locy)
    locx = plticker.MultipleLocator(base=1)
    ax.xaxis.set_major_locator(locx)
    
    ax.set_xlim([.2, 8])
    ax.set_ylim([200, 1300])

    ax.set_ylabel(r'$\sigma_v$ (km/s)', fontsize=20)
    ax.set_xlabel(r'$m_{200}$ ($10^{14}$M$_{\odot} h^{-1}$)', fontsize=20)
    ax.legend(loc='upper left', fontsize=15)
    for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(14)
    plt.show()


# ===================================================================================================
# =============================== galaxy-particle dispersion 1:1 ====================================
# ===================================================================================================


def galaxy_particle_disp_1to1(mask_magnitude = False, observed = True, include_stepCatalog = False):
    '''
    This function gathers the necessary data to compare the particle based dispersions and galaxy
    based dispersions of protoDC2 halos. Optionally, it also preforms halo matching across catalogs
    to compare the velocity dispersions of the upstream step catalog to that of the lightcone
    catalog.

    :param mask_magnitude: whether or not to preform a magnitude cut at -16 r-band
    :param observed: whether ot not to calculate the galaxy velocity dispersion as an observer would
                     find it (use the observed redshifts to get LOS velocities). If not, use 3d 
                     galay velocities
    :param include_stepCatalog: whether or not to match to the step catalog and include this data
    :return: None
    '''
    
    protoDC2 = h5py.File('data/protoDC2_clusters_shear_nocut_A.hdf5', 'r')
    obsSuff = ['1d', 'obs'][observed]
    halos = list(protoDC2.keys())
    n = len(halos)
    galDisp = np.zeros(n)
    sodDisp = np.zeros(n)
    if(include_stepCatalog):
        try:
            galDisp_steps = np.load('data/dispersion_1to1_snapshot_data_{}.npy'.format(obsSuff))
            bypass_snapshotLoop = 1
        except FileNotFoundError:
            print('loading snapshot catalog')
            protoDC2_steps = h5py.File('/media/luna1/jphollowed/protoDC2/'\
                                       'protoDC2_STEPS_clusters_nocut_A.hdf5', 'r')
            halos_steps = np.array([protoDC2_steps[h].attrs['fof_halo_tag'] 
                                    for h in list(protoDC2_steps.keys())])
            steps = np.array([protoDC2_steps[h].attrs['halo_step'] 
                              for h in list(protoDC2_steps.keys())])
            galDisp_steps = np.zeros(n)
            bypass_snapshotLoop = 0
    
    for j in range(n):
        halo = protoDC2[halos[j]]

        # get halo properties, including particle-based dispersion and mass
        zHost = halo.attrs['halo_z']
        zHost_err = halo.attrs['halo_z_err']
        sodDisp[j] = halo.attrs['sod_halo_vel_disp']
        if(observed): galDisp[j] = halo.attrs['gal_vel_disp_obs']
        else: galDisp[j] = halo.attrs['gal_vel_disp_1d']

        # if magnitude should be masked, dispersions must be recalculated with new populations
        # as given by a -16 r-band magnitude cut
        if(mask_magnitude):
            rMag = halo['magnitude:SDSS_r:rest'][:]
            rMag_mask = (rMag > -16)
            if(observed):
                # use redshifts of galaxies surviving the mask to calculate galaxy-based dispersion
                z = halo['redshiftObserver'][:][rMag_mask]
                pecV = ct.LOS_properVelocity(z, zHost)
                galDisp[j]  = stat.bDispersion(pecV)
            else:
                galDisp[j] = stat.dmDispersion(halo['vx'], halo['vy'], halo['vz'])

        # do halo matching across step catalog, measure peculiar LOS velocites, and dispersions
        if(include_stepCatalog and not bypass_snapshotLoop):
            print('matching to snapshot catalog for halo {}/{}'.format(j+1, n))
            lightcone_equivStep = halo.attrs['halo_step']
            step_mask =  (steps == lightcone_equivStep)
            halo_mask = (halos_steps == halo.attrs['fof_halo_tag'])
            mask = (halo_mask & step_mask)
            if(np.sum(mask) > 1): 
                raise ValueError('too many matches found for halo {} in step catalog (bad masking)'
                                 .format(halos[j]))
            halo_step = protoDC2_steps[np.array(list(protoDC2_steps.keys()))[mask][0]]

            if(observed):
                pecV_step = cpz.pecZ(halo_step['x'], halo_step['y'], halo_step['z'], halo_step['vx'],
                                     halo_step['vy'], halo_step['vz'], vPec_only=True)
                galDisp_steps[j] = stat.bDispersion(pecV_step)
            else:
                galDisp_steps[j] = stat.dmDispersion(halo_step['vx'], halo_step['vy'], halo_step['vz'])

            np.save('data/dispersion_1to1_snapshot_data_{}.npy'.format(obsSuff), galDisp_steps)

    if(include_stepCatalog): 
        plot_galaxy_particle_disp_1to1(galDisp, sodDisp, include_stepCatalog=True, 
                                       sigma_step = galDisp_steps)
    else: plot_galaxy_particle_disp_1to1(galDisp, sodDisp)

# -------------------------------------------------------------------------------------------------

def plot_galaxy_particle_disp_1to1(sigma, sigmadm, include_stepCatalog = False, sigma_step=None):
    '''
    Plot the particle_based vs glaxy-based velocity distribution for each halo as passed.
    This function is meant to be called by mass_vDisp_relation() above.

    :param sigma: An array of galaxy-based dispersions
    :param sigmadm: An array of particle-based dispersions
    :param include_stepCatalog: whether or not to include compairson data from the step catalog
    '''

    if(include_stepCatalog and sigma_step is None):
        raise ValueError('step catalog dipsersions must be passed if include_stepCatalog == True')
    
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(sigma, sigmadm, 'xb', ms=7, mew=1.6, alpha=0.5, label='lightcone data')
    if(include_stepCatalog):
        ax.plot(sigma_step, sigmadm, 'xg', ms=7, mew=1.6, alpha=0.5, label='snapshot data')
    ax.plot([200, 1800], [200, 1800], '--k', lw=2)

    ax.set_ylim([200, 1800])
    ax.set_xlim([200, 1800])
    ax.legend(loc='lower right')
    ax.grid()

    ax.set_ylabel(r'$\sigma_{v,\mathrm{particles}}$ (km/s)', fontsize=20)
    ax.set_xlabel(r'$\sigma_{v,\mathrm{galaxies}}$ (km/s)', fontsize=20)
    plt.show()


# ===================================================================================================
# ============================ velocity-component preference by color ===============================
# ===================================================================================================


def vel_component_preference(mask_var = 'gr', mask_type = 'cut', mask_magnitude = False):
    '''
    This function finds the radial and tangential velocity component of each protoDC2 galaxy, 
    in order to inspect the preference of red vs. blue galaxies. The idea is that blue galaxies should,
    if well-modeled, prefer radial velocities, whereas red galaxies are more virialized.

    :param mask_var: what galaxy property to preform the color cut on. Default is 'gr', 
                     meaning that the cut will be preformed in g-r space. Other valid 
                     options are 'sfr' (which will cut at the median of this value)
    :param mask_magnitude: whether or not to apply a cut in r-band magnitude space of -16
    :return: nothing
    '''

    protoDC2_stack = h5py.File('data/protoDC2_haloStack_shear_nocut_A.hdf5', 'r')
    g_band = protoDC2_stack['magnitude:SDSS_g:rest'][:]
    r_band = protoDC2_stack['magnitude:SDSS_r:rest'][:]
    gr_color = g_band - r_band
    sfr = protoDC2_stack['totalStarFormationRate'][:]

    # do galaxy cut
    if(mask_var == 'gr'):
        if(mask_type == 'cut'): 
            redMask = gr_color > 0.25
            blueMask = ~redMask
        elif(mask_type == 'percentile'):
            raise ValueError('cannot use mask_type \'percentile\' with mask_var \'gr\'')
    elif(mask_var == 'sfr'):
        if(mask_type == 'cut'): 
            redMask = sfr < np.median(sfr)
            blueMask = ~redMask
        elif(mask_type == 'percentile'): 
            redMask = sfr <= np.percentile(sfr, 15)
            blueMask = sfr >= np.percentile(sfr, 85)
    if(mask_magnitude):
        magMask = r_band > -16
    else:
        magMask = np.ones(len(r_band), dtype=bool)

    vMag = protoDC2_stack['vMag_normed'][:]
    vRad = protoDC2_stack['vRad_normed'][:]
    vTan = protoDC2_stack['vTan_normed'][:]
    radDist = protoDC2_stack['radDist_normed'][:]
    iMag = protoDC2_stack['magnitude:SDSS_i:rest'][:]
    sfr = protoDC2_stack['totalStarFormationRate'][:] / protoDC2_stack['totalMassStellar'][:]
    plot_vel_component_preference(radDist, vMag, vRad, vTan, gr_color, iMag, sfr, redMask, blueMask)

# -------------------------------------------------------------------------------------------------

def plot_vel_component_preference(r, v, vRad, vTan, gr_colors, i, sfr, redMask, blueMask):
    '''
    This function plots galaxy velocity component ratios as a histogram, in order to visually 
    inspect for component preference in red galaxies vs. blue galaxies. 
    '''
    
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    binVar = r

    n_bins = 100
    sortMask = np.argsort(binVar)
    rbins = np.array_split(binVar[sortMask], n_bins)
    vbins = np.array_split(vRad[sortMask], n_bins)
    avg_r = np.array([np.mean(rb) for rb in rbins])
    avg_v = np.array([np.mean(vb) for vb in vbins])
    err = np.array([np.std(vb)/np.sqrt(len(vb)) for vb in vbins])
    meanRedRad = np.median(r[redMask])
    meanBlueRad = np.median(r[blueMask])
    meanRedVel = np.median(vRad[redMask])
    meanBlueVel = np.median(vRad[blueMask])
    pdb.set_trace()

    if(not np.array_equal(binVar, sfr)):
        redHist = np.histogram2d(binVar[redMask], vRad[redMask], bins=40, normed=False) 
        blueHist = np.histogram2d(binVar[blueMask], vRad[blueMask], bins=40, normed=False) 
        pt.hist2d_to_contour(redHist, ax=ax,  log=True, cmap='Reds')    
        pt.hist2d_to_contour(blueHist, ax=ax, log=True, cmap='Blues')    

    ax.plot(avg_r, avg_v, '-om', ms=8, label='average')
    ax.plot(avg_r, avg_v, '-k', alpha=0.1)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], '--k', lw=2, label='zero')
    ax.fill_between(avg_r, avg_v + err, avg_v - err, color = [1, .8, .8])
    if(np.array_equal(binVar, sfr)):
        ax.set_xscale('log')
    plt.plot([0, 0], [0, 0], '-r', label='red gals')
    plt.plot([0, 0], [0, 0], '-b', label='blue gals')

    ax.grid()
    #ax.set_xlabel(r'$\mathrm{SFR} / M_{\mathrm{stellar}}$', fontsize=24)
    ax.set_xlabel(r'$\mathrm{r} / r_{200}$', fontsize=24)
    ax.set_ylabel(r'$v_{\mathrm{radial}}/\sigma_{v}$', fontsize=24)
    ax.legend(loc='upper right')
    ax.set_xlim([0, 3])
    ax.set_ylim([-6.2, 6.2])
    plt.show()


# ===================================================================================================
# =============================== visualize redshift-space distortion ===============================
# ===================================================================================================

def comp_halo_shapes():
    '''
    This function plots halos in 3 dimensions of RA vs Dec vs Mpc to show
    their 3d shapes, before and after redshift distortion
    '''
    pFile = 'protoDC2_clusters_full_shear_nocut_dust_elg_shear2_mod.hdf5'
    protoDC2 = h5py.File('/media/luna1/jphollowed/protoDC2/{}'.format(pFile), 'r')
    halosGroup = protoDC2['clusters']
    halos = [halosGroup[key] for key in list(halosGroup.keys())]

    for halo in halos:

        if(halo.name != '/clusters/halo_231'): continue
        ra = halo['ra'][:] / 60 / 60
        dec = halo['dec'][:] / 60 / 60
        x = halo['x']
        y = halo['y']
        z = halo['z']
        zHubb = halo['redshiftHubble'][:]
        zDist = halo['redshift'][:]
        print(halo)
        if(np.mean(zHubb) > 0.2): continue
        fig = plt.figure(1)
        fig.clf()
        plt.hold(True)
        ax1 = plt.subplot2grid([3,2], [0,0], rowspan=2, projection='3d')
        ax2 = plt.subplot2grid([3,2], [0,1], rowspan=2, projection='3d')
        ax1.scatter(cosmo.comoving_distance(zHubb), dec, ra, color=np.array([111,168,220])/255)
        ax2.scatter(cosmo.comoving_distance(zDist), dec, ra, color=np.array([111,168,220])/255)
        #ax1.scatter(z, dec, ra)
        ax1.set_xlim(ax2.get_xlim())
        ax1.set_ylim(ax2.get_ylim())
        ax1.set_zlim(ax2.get_zlim())
        for ax in [ax1, ax2]:
            ax.set_xlabel('comoving LOS distance (Mpc)', fontsize=14, labelpad=9)
            ax.set_ylabel('Dec (deg)', fontsize=14, labelpad=14)
            ax.set_zlabel('RA (deg)', fontsize=14, labelpad=14)

        ax3 = plt.subplot2grid([3,2], [2,0])
        ax4 = plt.subplot2grid([3,2], [2,1])
        ax3.hist(zHubb, 5, histtype='step', lw=2.1, color=np.array([246,178,107])/255)
        ax4.hist(zDist, 20, histtype='step', lw=2.1, color=np.array([246,178,107])/255)
        ax3.hist(zHubb, 5, lw=0, color=np.array([246,178,107])/255, alpha=0.4)
        ax4.hist(zDist, 20, lw=0, color=np.array([246,178,107])/255, alpha=0.4)
        ax4.set_xlim([min(zDist), max(zDist)])
        ax3.set_xlim(ax4.get_xlim())
        ax4.set_ylim(ax3.get_ylim())
        ax3.set_xlabel(r'$z$', fontsize=22)
        ax4.set_xlabel(r'$z$', fontsize=22)
        ax3.set_ylabel(r'$N$', fontsize=22)
        ax3.grid()
        ax4.grid()

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
