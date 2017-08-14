'''
Joe Hollowed
Last edited 2/10/17

Collection of visualization functions, meant to be used to plot output
of corresponding functions in stackAnalysis.py
'''

import matplotlib.pyplot as plt
import glob
import numpy as np
import pdb
import os
import h5py
import dtk
from astropy.constants import M_sun
from astropy.cosmology import WMAP7 as cosmo
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as plticker
import matplotlib.colors as colors


#======================================================================================================

def initPlot():
    '''
    configures pyplot
    '''

    plt.rc('text', usetex=True)
    params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
    plt.rcParams.update(params)


#======================================================================================================
    
def stack_segregation(catalog=0):
    '''
    Same as stack_segregation, but using binned stack data instead, as described in, and 
    output from, stack_Analysis.stack_binning()
    
    :param bin_prop: The core property used to bin the stack in stackAnalysis.stacked_binning()
    :param catalog: the core catalog to be used ('cut', 'merged', or 'unprocessed')
    '''

    initPlot()
    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    data_path = 'data/coreCatalog/{}/segregatedCores_processed_z1r0-5nc.hdf5'.format(catalogName)
    fig_path = '/home/jphollowed/figs/newSegrFigs/{}'.format(catalogName)
    data = h5py.File(data_path, 'r')

    step_groups = list(data.keys())
    props = ['infall_mass_segr', 'radius_segr', 'time_since_infall_segr']
    propNames = [r'$\mathrm{ \detokenize{infall_mass} }$', r'$\mathrm{radius}$', 
                 r'$\mathrm{ \detokenize{time_since_infall} }$']
    colors = ['m', 'g', 'r']

    zTool = dtk.StepZ(200, 0, 500)

    # loop through binned data and save a plot at each redshift
    for n in range(len(step_groups)):
        
        fig = plt.figure(1)
        fig.clf()
        step = int(step_groups[n].split('_')[-1])
        z = zTool.get_z(step)
        plt.suptitle('z = {}'.format(np.round(z, 3)), fontsize=18)
        
        for j in range(len(props)):
        
            step_data = data[step_groups[n]][props[j]]
            vDisp = step_data['bin_vDisp'][:]
            err = step_data['bin_vDispErr'][:]
            pcen = step_data['pcen_lowerPop'][:]
            
            ax1 = fig.add_subplot(3, 1, j+1)
            
            # plot mixed-bin dispersions
            ylim = [0.80, 1.1]
            xlim = [-0.05, 1.05]
            ax1.plot(xlim, np.ones(len(xlim)), '--', linewidth=1.2, color='gray')
            ax1.plot(pcen, vDisp, '.-', color = colors[j], mew = 0.2, mfc='black', lw=1.5, 
                     ms=8, label = propNames[j])
            ax1.fill_between(pcen, vDisp - err, vDisp + err, color=colors[j], alpha = 0.2)
            ax1.legend(fontsize=12, loc='lower right')

            ax1.set_xlabel(r'$\% (< \mathrm{median})$', fontsize=16)
            ax1.set_ylabel(r'$\sigma_v / \sigma_{v,\mathrm{all}}$', fontsize=16)
            ax1.set_ylim(ylim)    
            ax1.set_xlim(xlim)    
        plt.savefig('{}/{}.pdf'.format(fig_path, z))
        print('saved figure at z = {}'.format(z))
        #plt.show()


#======================================================================================================

def stack_binning(catalog=0):
    '''
    Same as stack_segregation, but using binned stack data instead, as described in, and 
    output from, stack_Analysis.stack_binning()
    
    :param bin_prop: The core property used to bin the stack in stackAnalysis.stacked_binning()
    :param catalog: the core catalog to be used ('cut', 'merged', or 'unprocessed')
    '''

    initPlot()
    catalogName = ['BLEVelocity', 'MedainVelocity', 'CentralVelocity'][catalog]
    data_path = 'data/coreCatalog/{}/segregatedCores_processed_z1e-06r3nc.hdf5'.format(catalogName)
    fig_path = 'data/coreCatalog/{}/'.format(catalogName)
    data = h5py.File(data_path, 'r')

    step_groups = list(data.keys())
    props = ['infall_mass_segr', 'radius_segr', 'time_since_infall_segr']
    colors = ['m', 'g', 'r']

    # loop through binned data and save a plot at each redshift
    for n in range(len(step_groups)):
        
        fig = plt.figure(1)
        fig.clf()
        
        for j in range(len(props)):
        
            step_data = data[step_groups[n]][props[j]]
            vDisp = step_data['bin_vDisp'][:]
            err = step_data['bin_vDispErr'][:]
            bin_avg = step_data['bin_center'][:]
            bin_width = step_data['bin_width'][:]
            
            ax1 = fig.add_subplot(3, 1, j+1)
            
            # plot binned dispersions
            ax1.plot(bin_avg, np.ones(len(bin_avg)), '--', linewidth=1.6, color='black')
            ax1.plot(bin_avg, vDisp, '.-', color = colors[j], mew = 0.2, mfc='black', lw=1.5, ms=8)
            ax1.fill_between(bin_avg, vDisp - err, vDisp + err, color=colors[j], alpha = 0.2)
            if(j==0): ax1.set_xscale('log')
            ax1.set_xlabel(r'$\mathrm{{ \detokenize{{ {} }} }}$'.format(props[j].split('segr')[0]), 
                           fontsize=18)
            ax1.set_ylabel(r'$\sigma_v / \sigma_{v,\mathrm{all}}$', fontsize=18)
            #ax1.set_ylim([0.90, 1.10])    
            ax1.text(0.05, 0.78, r'$\mathrm{{ \detokenize{{ {} }} }}$'
                                   .format(step_groups[n]),transform=ax1.transAxes, fontsize = 14)
            #plt.savefig('{}/{}.png'.format(fig_path, z))
            #print('saved figure at z = {}'.format(z))
        plt.show()


#======================================================================================================
def double_segregation(catalog=0):
    '''
    Same as stack_segregation, but using binned stack data instead, as described in, and 
    output from, stack_Analysis.stack_binning()
    
    :param bin_prop: The core property used to bin the stack in stackAnalysis.stacked_binning()
    :param catalog: the core catalog to be used ('cut', 'merged', or 'unprocessed')
    '''

    initPlot()
    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    datadir = 'data/coreCatalog/{}'.format(catalogName)
    data_path = '{}/segregatedCores_double_processed_z1r0-5nc.hdf5'.format(datadir)
    fig_path = '/home/jphollowed/figs/newSegrFigs/{}_double'.format(catalogName)
    data = h5py.File(data_path, 'r')

    step_groups = list(data.keys())
    props = ['radius_segr', 'time_since_infall_segr']
    vary_prop = ['vary_infall_mass_1', 'vary_infall_mass_2']

    propNames = ['\mathrm{{ radius }}', '\mathrm{{ \detokenize{{time_since_infall}} }}']
    colors = ['g', 'r']
    lines = ['.-', '.--']
    vary = ['<', '>']

    zTool = dtk.StepZ(200, 0, 500)

    # loop through binned data and save a plot at each redshift
    for n in range(len(step_groups)):
        
        fig = plt.figure(1)
        fig.clf()
        step = int(step_groups[n].split('_')[-1])
        z = zTool.get_z(step)
        plt.suptitle(r'$z = {}$'.format(np.round(z, 3)), fontsize=18)
        
        for j in range(len(props)):
            for k in range(len(vary_prop)):
                step_data = data[step_groups[n]][props[j]][vary_prop[k]]
                vDisp = step_data['bin_vDisp'][:]
                err = step_data['bin_vDispErr'][:]
                pcen = step_data['pcen_lowerPop'][:]
                
                ax1 = fig.add_subplot(2, 1, j+1)
                
                # plot mixed-bin dispersions
                ylim = [0.80, 1.1]
                xlim = [-0.05, 1.05]
                ax1.plot(xlim, np.ones(len(xlim)), '--', linewidth=1.2, color='gray')
                ax1.plot(pcen, vDisp, lines[k], color = colors[j], mew = 0.2, mfc='black', lw=1.5, 
                         ms=8, label = r'${} \>\>\mathrm{{ ({} median\>\>\detokenize{{infall_mass)}} }}$'.format(propNames[j], vary[k]))
                ax1.fill_between(pcen, vDisp - err, vDisp + err, color=colors[j], alpha = 0.2)
                ax1.legend(fontsize=12, loc='lower right')

                ax1.set_xlabel(r'$\% (< \mathrm{median})$', fontsize=16)
                ax1.set_ylabel(r'$\sigma_v / \sigma_{v,\mathrm{all}}$', fontsize=16)
                ax1.set_ylim(ylim)    
                ax1.set_xlim(xlim)    
        plt.savefig('{}/{}.pdf'.format(fig_path, z))
        print('saved figure at z = {}'.format(z))
        #plt.show()


#======================================================================================================

def VvsR_particlesVcores():
    '''
    Plots the binned average radial velocity vs radial distance 
    as output from stackAnalysis.V_vs_R() script on Jupiter and Datastar for
    particles and cores, respectively
    '''
    
    fig_path = '/home/jphollowed/figs/VvsR_coresVparticles_figs/merged_figs'    
    particleAvgs_path = ('/home/jphollowed/data/hacc/alphaQ/haloParticles/' \
                 'stackAnalysis/VvsR_particles/')
    coreAvgs_path = '/home/jphollowed/data/hacc/alphaQ/coreCatalog_cut/stackAnalysis/VvsR_cores/'
    particleAvgs_files = sorted(glob.glob('{}/VvsR_particles*'.format(particleAvgs_path)))
    coreAvgs_files = sorted(glob.glob('{}/VvsR_cores*'.format(coreAvgs_path)))
    initPlot()
    
    # loop through V_vs_R data and save plot at each redshift 
    for n in range(len(particleAvgs_files)):
        
        coreAvgs = np.load(coreAvgs_files[n])
        particleAvgs = np.load(particleAvgs_files[n])
        z = coreAvgs_files[n].split('_')[-1].rsplit('.',1)[0]

        core_avg_r = coreAvgs['avg_r']
        core_avg_v = coreAvgs['avg_v']
        core_err = coreAvgs['error']
        
        particle_avg_r = particleAvgs['avg_r']
        particle_avg_v = particleAvgs['avg_v']
        particle_err = particleAvgs['error']
    
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
    
        ax1.plot(core_avg_r, core_avg_v, '+r', ms=10, mew=1.5, label='bin avg core velocity')
        ax1.fill_between(core_avg_r, core_avg_v-core_err, core_avg_v+core_err, 
                 color=[1, .7, .7])
        
        ax1.plot(particle_avg_r, particle_avg_v, '+b', ms=10, mew=1.5, 
             label='bin avg particle velocity')
        ax1.fill_between(particle_avg_r, particle_avg_v-particle_err, 
                 particle_avg_v+particle_err, color=[.7, .7, 1])
        
        ax1.plot([0, 2.5], [0, 0], '--', color='black', linewidth=1.5)
        ax1.set_xlabel(r'$r/r_{200}$', fontsize=24)
        ax1.set_ylabel(r'$\bar{v}_{radial}/\sigma_{v,3D}$', fontsize=24)
        ax1.set_ylim([-4, 4])
        ax1.set_xlim([0, 2.5])
        ax1.legend()

        ax1.text(0.05, 0.85, 'z = {}'.format(z),transform=ax1.transAxes, fontsize = 16)

        plt.savefig('{}/{}.png'.format(fig_path, z))
        print('saved figure for z = {}'.format(z))


#======================================================================================================

def vHistograms(allCats = True, catalog = 0, cutR = False, processed = True, norm=False):

    initPlot()
    if not allCats:
        catalogName = [['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]]
    else:
        catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity']
    processPrefix = ['un', ''][processed]
    corePath = ['data/coreCatalog/{}'.format(cat) for cat in catalogName]
    stackFile = [h5py.File('{}/stackedCores_{}processed.hdf5'.format(path, processPrefix), 'r+')
                 for path in corePath]

    vel = [stack['step_499']['v_coreNorm'][:] for stack in stackFile]
    lw = [1, 1.2, 1.2]
    c = [[0.3, 0.3, 1], [0, 1, 0], [1, .3, 0.3]]
    bins = np.linspace(0, 8, 75)
    if(allCats == False): ht = 'stepfilled'
    else: ht = 'step'

    if(cutR):
        rad = [stack['step_499']['r_norm'][:] for stack in stackFile]
        mask = [(r > 1e-1) for r in rad]
        imask = [(r < 1e-1) for r in rad]
        vel_cut = [v[mask] for v in vel]
        vel_nocut = [v[imask] for v in vel]

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.hold(True)

    for i in range(len(vel)):
        if(cutR):
            color = c[catalog]
            ax.hist(vel_nocut[i], bins, histtype=ht, lw=lw[i], color=color, 
                    label=catalogName[i] + '(r $<$ 0.1 r200)', normed=norm, alpha = 0.5)
            ax.hist(vel_cut[i], bins, histtype=ht, lw=lw[i], color=color, 
                    label=catalogName[i] + '(r $>$ 0.1 r200)',normed=norm, alpha = 0.8)
        else:
            ax.hist(vel[i], bins, histtype=ht, lw=lw[i], color=c[i], label=catalogName[i], normed=norm)
    
    ax.legend(fontsize=14)
    ax.set_xlabel(r'$v/\sigma_{v,\mathrm{cores}}$', fontsize=18)
    #ax.set_ylabel(r'$\mathrm{pdf}$', fontsize=18)

    plt.show()


#======================================================================================================

def plot_sigVm(catalog=0, processed=True, scatter = True, maskR = True):

    initPlot()
    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    processSuffix = ['un', ''][processed]

    fig_path = '/home/jphollowed/figs/dispVmass_figs'
    halo_path = ('data/coreCatalog/{}/haloCores_{}processed_masked.hdf5'
                 .format(catalogName, processSuffix))
    if maskR:
        coreVelDisp = 'core_vel_disp_masked'
        coreVelErr = 'core_vel_disp_err_masked'
    else:   
        coreVelDisp = 'core_vel_disp'
        coreVelErr = 'core_vel_disp_err'
    
    stepZ = dtk.StepZ(200, 0, 500)

    alphaQ = h5py.File(halo_path, 'r')
    steps = np.array(list(alphaQ.keys()))
    zs = np.array([stepZ.get_z(int(step.split('_')[-1])) for step in steps])
    sortOrder = np.argsort(zs)
    zs = zs[sortOrder]
    steps = steps[sortOrder]
    halo_masses = []
    halo_vDisp = []
    core_vDisp = []
    core_vDisp_err = []
    core_counts = []

    for j in range(len(zs)):
        z = zs[j]
        a = cosmo.scale_factor(z)
        h = cosmo.H(z).value/100
        thisStep = alphaQ[steps[j]]
        halo_tags = thisStep.keys()
        halos = [thisStep[tag].attrs for tag in halo_tags]

        halo_masses = np.concatenate([halo_masses, [halo['sod_halo_mass']*h for halo in halos]])
        halo_vDisp = np.concatenate([halo_vDisp, [halo['sod_halo_vel_disp']*a for halo in halos]])
        core_vDisp = np.concatenate([core_vDisp, [halo[coreVelDisp]*a for halo in halos]])
        core_vDisp_err = np.concatenate([core_vDisp_err, 
                                        [halo[coreVelErr]*a for halo in halos]])
        core_counts = np.concatenate([core_counts, 
                                     [thisStep[tag]['core_tag'].size for tag in halo_tags]]).astype(int)
        
        badMask = np.logical_or(np.isnan(core_vDisp), np.isnan(core_vDisp_err))
        core_vDisp = core_vDisp[~badMask]
        core_vDisp_err = core_vDisp_err[~badMask]
        core_haloMasses = halo_masses[~badMask]

    # Overplotting Evrard relation
    t_x = np.linspace(4.5e13, 2e15, 300)
    sig_dm15 = 1082.9
    sig_dm15_err = 4
    alpha = 0.3361
    alpha_err = 0.0026
    t_y = sig_dm15 * (t_x / 1e15)**alpha
    t_y_high = (sig_dm15 + sig_dm15_err) * (t_x / 1e15)**(alpha + alpha_err)
    t_y_low = (sig_dm15 - sig_dm15_err) * (t_x/ 1e15)**(alpha - alpha_err)
    
    # preforming Least Squares on AlphaQuadrant Data
    # (fitting to log linear form, as in Evrard et al.)
    # X = feature matrix (masses)
    # P = parameter matrix (sig_dm15(log intercept) and alpha(log slope))
    X = np.array([np.log(mi / 1e15) for mi in halo_masses])
    X = np.vstack([X, np.ones(len(X))]).T
    P = np.linalg.lstsq(X, np.log(halo_vDisp))[0]
    alpha_fit = P[0]
    sig_dm15_fit = np.e**P[1]

    # preforming Least Squares on core data
    X = np.array([np.log(mi / 1e15) for mi in core_haloMasses])
    X = np.vstack([X, np.ones(len(X))]).T
    P = np.linalg.lstsq(X, np.log(core_vDisp))[0]
    alpha_fit_cores = P[0]
    sig_dm15_fit_cores = np.e**P[1]

    fit_x = np.linspace(4.5e13, 2e15, 300)
    fit_y = sig_dm15_fit * (( (fit_x) / (1e15) )**alpha_fit)
    fit_y_cores = sig_dm15_fit_cores * (( (fit_x) / (1e15) )**alpha_fit_cores)
    
    print('Dm fit: sig_dm15 = {}, alpha = {}'.format(sig_dm15_fit, alpha_fit))
    print('Core fit: sig_dm15 = {}, alpha = {}'.format(sig_dm15_fit_cores, alpha_fit_cores))
    print('Bias = sig_cors / sig_DM = {}'.format(sig_dm15_fit_cores/sig_dm15_fit))

    # ---------- plotting ----------
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    vRange = [200, 1550]
    mRange = [4.5e13, 2e15]
    #plt.hold(True)
    colorVar = core_vDisp_err

    if(scatter):
        p = ax.loglog(halo_masses, halo_vDisp, '^', markersize=8, color='black', zorder=1, 
                  label='DM dispersion')
        p_cores = ax.scatter(core_haloMasses, core_vDisp, lw=0, c=colorVar, 
                     zorder=2,  label='Core dispersion', s=10, 
                     norm=colors.LogNorm(vmin=10, vmax=max(colorVar)))
        cbar=plt.colorbar(p_cores, ticks=np.linspace(0, max(colorVar), 21), extend='max')
        cbar.ax.set_yticklabels([str(int(i)) for i in np.linspace(0, max(colorVar), 21)])
    else:
        nBins = 30
        xRange = np.log10(mRange)
        yRange = np.log10(vRange)
        xbins = np.logspace(xRange[0], xRange[1], nBins)
        ybins = np.logspace(yRange[0], yRange[1], nBins)
        bins = [xbins, ybins]

        xCenters = xbins[:-1] + np.diff(xbins) / 2
        yCenters = ybins[:-1] + np.diff(ybins) / 2
        X, Y = np.meshgrid(xCenters, yCenters)
        
        haloDensity = np.histogram2d(halo_masses, halo_vDisp, bins)
        coreDensity = np.histogram2d(halo_masses, core_vDisp, bins) 
        ax.contourf(X, Y, haloDensity[0].T, cmap='Greys', vmin=0, label='core dispersion')
        ax.contourf(X, Y, coreDensity[0].T, cmap='PuBu', vmin=0, label='particle dispersion')

    t = ax.loglog(t_x, t_y, '--b', linewidth = 1.2)
    t_err = plt.fill_between(t_x, t_y_low, t_y_high, color=[0.7, 0.7, 1])
    fit = ax.loglog(fit_x, fit_y, 'b', linewidth = 1.2)
    fit_cores = ax.loglog(fit_x, fit_y_cores, 'g', linewidth = 1.2)

    ax.set_ylim(vRange)
    ax.set_xlim(mRange)
    
    ax.set_ylabel(r'$\sigma_v$', fontsize=16)
    ax.set_xlabel(r'$h(z)\mathrm{M}_{200}$', fontsize=16)
    ax.legend(loc=4)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    loc = plticker.MultipleLocator(base=100)
    ax.yaxis.set_major_locator(loc)
    #plt.text(0.05, 0.8, 'Dm fit: sig_dm15 = {:.2f}, alpha = {:.2f}'
    #              .format(sig_dm15_fit, alpha_fit),
    #     transform=ax.transAxes, fontsize = 14)
    #plt.text(0.05, 0.75,'Core fit: sig_dm15 = {:.2f}, alpha = {:.2f}'
    #              .format(sig_dm15_fit_cores, 
    #     alpha_fit_cores),transform=ax.transAxes, fontsize = 14)
    #plt.text(0.05, 0.7, 'Bias = sig_cors / sig_DM = {:.2f}'
    #     .format(sig_dm15_fit_cores/sig_dm15_fit),
    #     transform=ax.transAxes, fontsize = 14)
    plt.text(0.05, 0.8, '{}'.format(catalogName),transform=ax.transAxes, fontsize = 14)
        
    #plt.savefig('{}/{}_members.png'.format(fig_path, minN))
    plt.show()
