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
#==============================================================================================

def initPlot():
	'''
	configures pyplot
	'''

	plt.rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
	plt.rcParams.update(params)


#==============================================================================================

def stack_binning(bin_prop, catalog='cut'):
	'''
	Same as stack_segregation, but using binned stack data instead, as described in, and 
	output from, stack_Analysis.stack_binning()
	
	:param bin_prop: The core property used to bin the stack in stackAnalysis.stacked_binning()
	:param catalog: the core catalog to be used ('cut', 'merged', or 'unprocessed')
	'''

	initPlot()
	fig_path = '/home/jphollowed/figs/{}_binned_figs/figs_{}'.format(bin_prop,catalog)
	fig_path_1d = '/home/jphollowed/figs/{}_binned_1d/figs_{}'.format(bin_prop,catalog)
	data_path = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/stackAnalysis/'\
		     '{}_binned_cores'.format(catalog, bin_prop))
	full_stack_path = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/stackedHalos/by_redshift'
			   .format(catalog))

	data_files = sorted(glob.glob('{}/*binned_[0-9].*'.format(data_path)))
	full_stack_files = sorted(glob.glob('{}/*.npy'.format(full_stack_path)))
	if(len(data_files) == 0): 
		raise ValueError('No data found for property {}'.format(bin_prop))
	for path in [fig_path, fig_path_1d]:
		if not os.path.exists(path): os.makedirs(path)
	
	# loop through binned data and save a plot at each redshift
	for n in range(len(data_files)):
		
		step_data = np.load(data_files[n])
		full_stack_data = np.load(full_stack_files[n])
		vDisp = step_data['vDisp']
		err = step_data['vDisp_err']
		bin_avg = step_data['bin_avg']
		bin_width = step_data['bin_width']
		z = data_files[n].split('_')[-1].rsplit('.',1)[0]
		N_halos = full_stack_files[n].split('_')[-1].split('h')[0]
		N_cores = len(full_stack_data)
	
		fig, ax1 = plt.subplots(1)
		
		# plot binned dispersions
		ax1.plot(bin_avg, np.ones(len(bin_avg)), '--', linewidth=1.6, color='black')
		ax1.errorbar(bin_avg, vDisp, xerr=bin_width, yerr=None, 
			     color=[238/255, 130/255, 238/155], markerfacecolor='black',
			     linewidth=2, ms=15)
		ax1.set_xscale('log')
		ax1.set_xlabel(r'${}$'.format(bin_prop), fontsize=22)
		ax1.set_ylabel(r'$\sigma_v / \sigma_{v,\text{all}}$', fontsize=24)
		#ax1.set_ylim([0.90, 1.10])	
		ax1.text(0.05, 0.78, r'z = {}\\\\halos = {}\\\\cores = {}'
			 .format(z, N_halos, N_cores),transform=ax1.transAxes, fontsize = 20)
		plt.savefig('{}/{}.png'.format(fig_path, z))
		print('saved figure at z = {}'.format(z))


#==============================================================================================
	
def stack_segregation(segr_prop='infall_mass', max_r = 1, catalog='cut', vel_type = '', 
		      norm='_coreNorm', all_zs=False, particles=False):
	'''
	Plots the velocity segregation of the stacked Halo as output from 
	stackAnalysis.stacked_segregation(), on the property given as the first parameter
	(must match an exact column name of the core catalog tacked halo)

	:param segr_prop: The core property used to segregate velocities in 
			  stackAnalysis.stacked_segregation()
	:param max_r: maxmimum normalized radial distance of which to draw cores from
	:param catalog: the core catalog to be used ('cut', 'merged', or 'unprocessed')
	:param all_zs: whether ot not to use the full stack (all redshifts)
	:param particles: whether or not to use the particle rather than core data
	'''

	initPlot()
	
	if(particles):
		fig_path = ('/home/jphollowed/figs/{}_segregation_figs_maxR/figs_particles'
			    .format(segr_prop))
		fig_path_1d = ('/home/jphollowed/figs/{}_segregation_1d/figs_particles'
			       .format(segr_prop))
		data_path = ('/home/jphollowed/data/hacc/alphaQ/haloParticles/stackAnalysis/'\
		     	     '{}_segregation_particles'.format(segr_prop))
		stack_path = ('/home/jphollowed/data/hacc/alphaQ/haloParticles/stackedHalos/by_{}_segr'
		      	      .format(catalog, segr_prop))
		full_stack_path = ('/home/jphollowed/data/hacc/alphaQ/haloParticles/stackedHalos'\
				   '/by_redshift')
	else:
		fig_path = ('/home/jphollowed/figs/{}_segr_figs/figs_{}_{}{}'
			    .format(segr_prop,catalog, vel_type, norm))
		fig_path_1d = ('/home/jphollowed/figs/{}_segr_1d/figs_{}_{}{}'
			       .format(segr_prop,catalog, vel_type, norm))
		data_path = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/stackAnalysis/'\
		     	     '{}_{}{}_segr_cores'.format(catalog, segr_prop, vel_type, norm))
		stack_path = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/stackedHalos/'\
			      'by_{}_segr_{}{}'.format(catalog, segr_prop, vel_type, norm))
		full_stack_path = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/stackedHalos'\
				   '/by_redshift'.format(catalog))

	for path in [fig_path, fig_path_1d]:
		if not os.path.exists(path): os.makedirs(path)
		
	if(all_zs):
		data_files = sorted(glob.glob('{}/vSegr_all_zs.npy'.format(data_path)))
		# data_files_1d = sorted(glob.glob('{}/*Segregation_1d_full*'.format(data_path)))
		full_stack_files = sorted(glob.glob('{}/*full*.npy'.format(full_stack_path)))
	else:
		data_files = sorted(glob.glob('{}/*Segregation_[0-9].*'.format(data_path)))
		data_files_1d = sorted(glob.glob('{}/*Segregation_1d_[0-9]*'.format(data_path)))
		full_stack_files = sorted(glob.glob('{}/*.npy'.format(full_stack_path)))
	
	# loop through mass_segregation data and save a plot at each redshift
	for n in range(len(data_files)):
		
		step_data = np.load(data_files[n])
		step_data_1d = np.load(data_files[n])
		full_stack_data = np.load(full_stack_files[n])
		r_mask = full_stack_data['r_rad_norm'] <= max_r
		full_stack_data = full_stack_data[r_mask]

		vDisp = step_data['bin_vDisp']
		err = step_data['bin_vDisp_err']
		pcen = step_data['pcen']
		z = data_files[n].split('_')[-1].rsplit('.',1)[0]
		N_halos = full_stack_files[n].split('_')[-1].split('h')[0]
		N_cores = len(full_stack_data)
	
		fig, ax1 = plt.subplots(1)
		
		# plot mass segregation curve
		ax1.plot(pcen, np.ones(len(pcen)), '--', linewidth=1.6, color='black')
		ax1.plot(pcen, vDisp, '.-', color='g',
			 markerfacecolor='black',linewidth=2, ms=15)
		ax1.fill_between(pcen, vDisp + err, vDisp - err, color=[.85, 1, .85])
		ax1.set_xlabel(r'\% high \verb|{p}|'.format(p=segr_prop), fontsize=22)
		ax1.set_ylabel(r'$\sigma_v / \sigma_{v,\text{all}}$', fontsize=24)
		ax1.set_ylim([min(vDisp)-0.04, max(vDisp) + 0.04])	
		ax1.text(0.05, 0.78, r'z = {}\\\\halos = {}\\\\cores = {}'
			 .format(z, N_halos, N_cores),transform=ax1.transAxes, fontsize = 20)
		plt.savefig('{}/{}.png'.format(fig_path, z))
		print('saved figure at z = {}'.format(z))
	

#==============================================================================================

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


#==============================================================================================

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


#==============================================================================================

def plot_sigVm(catalog=0, processed=True, scatter = False):

    initPlot()
    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    processSuffix = ['un', ''][processed]

    fig_path = '/home/jphollowed/figs/dispVmass_figs'
    halo_path = ('data/coreCatalog/{}/haloCores_{}processed.hdf5'
             .format(catalogName, processSuffix))
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
        core_vDisp = np.concatenate([core_vDisp, [halo['core_vel_disp']*a for halo in halos]])
        core_vDisp_err = np.concatenate([core_vDisp_err, 
                                        [halo['core_vel_disp_err']*a for halo in halos]])
        core_counts = np.concatenate([core_counts, 
                                     [thisStep[tag]['core_tag'].size for tag in halo_tags]])
         
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
    X = np.array([np.log(mi / 1e15) for mi in halo_masses])
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
    plt.hold(True)
    
    if(scatter):
        p = ax.loglog(halo_masses, halo_vDisp, '^', markersize=8, color='black', zorder=1, 
                  label='DM dispersion')
        p_cores = ax.scatter(halo_masses, core_vDisp, lw=0, c=core_counts, 
                     zorder=2, norm=colors.LogNorm(vmin=10, vmax=200), 
                     label='Core dispersion', s=10, cmap='PuBuGn')
        cbar=plt.colorbar(p_cores, ticks=np.linspace(0, 200, 21), extend='max')
        cbar.ax.set_yticklabels([str(int(i)) for i in np.linspace(0, 200, 21)])
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
