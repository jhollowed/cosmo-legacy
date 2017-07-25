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

def initPlot():
	'''
	configures pyplot
	'''

	plt.rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
	plt.rcParams.update(params)

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
