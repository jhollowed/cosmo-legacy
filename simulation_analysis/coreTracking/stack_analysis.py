'''
Joe Hollowed
Last edited 2/7/2017

A collection of functions to do some analysis on the stacked halos output from 
stackedCores.py
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import dispersionStats as stat
import coreTools
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rcfn
import pdb
import glob
import os

def stack_binning(bin_prop, n_bins = 10, catalog='cut'):
	'''
	The description of this function is written in relation to the description of the
	stack_segregation() function:

	Similar the the stack_segregation() function, where I simply bin the stacked halos by
	some core property rather than segregating by it (i.e. instead of collecting two distinct
	populations and drawing from each in varying percentage, I just collect n_bins distinct
	populations, and measure the dispersion of each.

	I still use the same bootstrap method for finding the dispersion and dispersion error
	for each distribution (bin of cores). This can also all be done using 1D or 3D velocities.

	:param bin_prop: the core property to use to bin the stacked halo (str)
	:param n_bins: number of bins to use (number of distinct distributions to measure
		       dispersion on) (int)
	:param catalog: version of the core catalog to use ('unprocessed', 'cut', or 'merged') (str)
	'''

	save_dest = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/stackAnalysis/'\
			 '{}_binned_cores'.format(catalog, bin_prop))
	if not os.path.exists(save_dest): os.makedirs(save_dest)
	stack_path = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/stackedHalos/by_redshift'
		      .format(catalog))
	stacks = sorted(glob.glob('{}/stack_[0-9]*_[0-9]*'.format(stack_path)))
	stack_zs = [s.split('_')[-2].split('_')[-1] for s in stacks]
	
	# save plot data for each redshift
	for n  in range(len(stacks)):

		stack = np.load(stacks[n])
		z = stack_zs[n]

		print('\n\n----- Working on stack at z = {} ({}/{})-----'.format(z, n, len(stacks)-1)) 
		# get dispersion and center of the full stack, before binning		
		vDisp_all = stat.bDispersion(stack['v_coreNorm'])
		vAvg_all = stat.bAverage(stack['v_coreNorm'])
		vDisp_all_1d = stat.bDispersion(stack['v_1d_coreNorm'])
		vAvg_1d = stat.bDispersion(stack['v_1d_coreNorm'])
		
		# bin stack into bins of equal sample size on the specified core property
		prop = stack[bin_prop]
		bin_edges = np.percentile(prop, np.linspace(0, 100, n_bins).tolist())
		avg_prop = [(bin_edges[n+1] + bin_edges[n]) / 2 for n in range(len(bin_edges)-1)]	
		bin_widths = [bin_edges[n+1]-bin_edges[n] for n in range(len(bin_edges)-1)]

		# gather velocity data. In arrays below, 1st row is 1d data, 2nd row is 3d data
		vs_3d = stack['v_coreNorm']
		vs_1d = stack['v_1d_coreNorm']
		vs = np.vstack((vs_1d, vs_3d))
		vel_disp = np.empty((2,len(avg_prop)))
		vel_disp_err = np.empty((2,len(avg_prop)))
		
		# find dispersion for each bin
		for i in range(len(bin_widths)):
			if(i%5==0):print('working on bin {}'.format(i))
			mask = (prop <= bin_edges[i+1]) & (prop > bin_edges[i])
			bin_vs = np.array([vi[mask] for vi in vs])
			
			# do bootstrap resampling 
			# (first row is 1d dispersions, 2nd row is 3d dispersions)
			bootstrap_disp = np.zeros((2,1000))
			
			for j in range(1000): 
				if(j%500 == 0): print('bootstrap resample {}/{}'.format(j, 1000))
				next_sample_1d = np.random.choice(vs[0], size=len(vs[0]), replace=True)
				next_sample_3d = np.random.choice(vs[1], size=len(vs[1]), replace=True)
				bootstrap_disp[:,j] = [stat.bDispersion(next_sample_1d),
						       stat.bDispersion(next_sample_3d)] / vDisp_all
			vel_disp[:,i] = np.mean(bootstrap_disp, axis=1)
			vel_disp_err[:,i] = np.std(bootstrap_disp, axis=1) 
	
		cols = ['bin_avg', 'bin_width', 'vDisp', 'vDisp_err', 'vDisp_1d', 'vDisp_err_1d']
		output = np.rec.fromarrays([avg_prop, bin_widths, vel_disp[1], vel_disp_err[1], 
					    vel_disp[0], vel_disp_err[0]], names=cols) 
		np.save('{}/core_{}_binned_{}'.format(save_dest, bin_prop, z), output)


		
def stack_segregation(segr_prop= 'infall_mass', n_bins=21, max_r=1, max_z=10, catalog='cut', 
		      all_zs=False, have_stacks=False, particles=False, vel_type='', 
		      norm='_coreNorm'):
	'''
	Caclulate the segregation in velocity dispersion of cores of high and low 
	infall mass by drawing from the full stacked halo ("full" as in, across all reshifts). 
	
	This is done just as the approach described in Bayliss et al. 2016, 
	where we asign some mass cut defining the boundary between "high infall mass" and 
	"low infall mass", and then vary the fraction of cores that we draw from each of 
	those populations, and calculate the velocity dispersion of each "mixed mass" bin. 
	The first bin, then, will be 100% populated by low infall mass cores, the center bin
	populated by 50% low and 50% high infall mass cores, and the last bin populated 
	100% by high infall mass cores.

	This entire procedure is done using both 1D and 3D velocities/dispersions.
	
	2/23/17: I have updated ths function to segregate, as described above, on any core property, 
		 not just infall mass. Specify the data column (from core catalog) desired to be used
		 for segregation with the 'segr_prop' parameter

	:param segr_prop: the core property on which to calculate velocity segregation
			  (must match an exact column name of the core catalog data) 
	:param n_bins: the number of mixed mass bins to include in the analysis (default = 21)
	:param max_r: maximum normalized radial distance of which to draw cores from 
	:param catalog: version of core catalog to use ('unprocessed', 'cut', or 'merged')
	:param all_zs: whether or not to use the full stack (all redshifts)
	:param have_stacks: whether or not to use saved segregation stacks or make new ones
			   (essentially, whether or not you have run this function for a 
			    given core property yet. If True, then do no create new segregation stacks)
	:param particles: if true, preform on stacked particles rather than cores
	:return: None
	'''

	particle_path = '/home/jphollowed/data/hacc/alphaQ/haloParticles'
	core_path = '/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}'.format(catalog)
	if(particles):
		stack_save_dest =  ('{}/stackedHalos/by_{}_segr'.format(particle_path, segr_prop))
		data_save_dest = ('{}/stackAnalysis/'\
				 '{}_segregation_particles'.format(particle_path, segr_prop))
		stack_path = ('{}/stackedHalos/by_redshift'.format(particle_path))
		normalize = 'n'
	else:
		stack_save_dest =  ('{}/stackedHalos/by_{}_segr_{}{}'.format(core_path, 
								      segr_prop, vel_type, norm))
		data_save_dest = ('{}/stackAnalysis/{}_{}{}_segr_cores'.format(core_path, 
									segr_prop, vel_type, norm))
		stack_path = ('{}/stackedHalos/by_redshift'.format(core_path))
		print('Using {} velocity with {} normalization'.format(vel_type, norm))
	
	if(all_zs): 
		stacks = glob.glob('{}/stack_full*'.format(stack_path))
		stack_zs = ['all_zs']
	else: 
		stacks = sorted(glob.glob('{}/stackedParticles_[0-9]*_[0-9]*'.format(stack_path)))
		stack_zs = [s.split('_')[-3].split('_')[-1] for s in stacks]
	
	if not os.path.exists(data_save_dest): os.makedirs(data_save_dest)
	if not os.path.exists(stack_save_dest): os.makedirs(stack_save_dest)

	if not all_zs:
		z_mask = np.array([float(z) for z in stack_zs]) <= max_z
		stacks = np.array(stacks)[z_mask]
		stack_zs = np.array(stack_zs)[z_mask]
	

	for n  in range(len(stacks)):
		stack = np.load(stacks[n])
		stack = np.random.choice(stack, size=50000, replace=False) # get rid of this
		r_mask = stack['r_rad_norm'] <= max_r
		stack = stack[r_mask]

		# get dispersion and center of the full stack, before segregation
		vDisp_all = stat.bDispersion(stack['v{}{}'.format(vel_type, norm)])
		vAvg_all = stat.bAverage(stack['v{}{}'.format(vel_type, norm)])
		#vDisp_all_1d = stat.bDispersion(stack['v_1d_{}orm'.format(normalize)])
		#vAvg_1d = stat.bDispersion(stack['v_1d_{}orm'.format(normalize)])
		z = stack_zs[n]
		
		# create vector of fractions according to n_bins
		fracs = np.linspace(0, 1, n_bins)
		bin_dispersions = np.zeros(n_bins)
		bin_disp_errors = np.zeros(n_bins)
		#bin_dispersions_1d = np.zeros(n_bins)
		#bin_disp_errors_1d = np.zeros(n_bins)
		print('\n\n----- Working on stack at z = {} ({}/{})-----'
		      .format(z, n, len(stacks))) 
	
		if not have_stacks:

			# segregate cores into low or high distribution on a halo-by-halo basis
			stack_upper = np.array([], dtype=stack.dtype).reshape(0, len(stack[0]))
			stack_lower = np.array([], dtype=stack.dtype).reshape(0, len(stack[0]))
			halo_tags = np.unique(stack['fof_halo_tag'])

			for halo_tag in halo_tags:
				halo = stack[stack['fof_halo_tag'] == halo_tag]
				segr_data = halo[segr_prop]
				prop_threshold = np.median(segr_data)
				halo_lower = halo[segr_data <= prop_threshold]
				halo_upper = halo[segr_data > prop_threshold]
				stack_lower = rcfn.stack_arrays([stack_lower, halo_lower], 
								usemask=False) 
				stack_upper = rcfn.stack_arrays([stack_upper, halo_upper], 
								usemask=False)

			N_lower = len(stack_lower)
			N_upper = len(stack_upper)

			# the number of cores used in the segregation analysis, 
			# N_tot should not exceed the minimum of these two populations
			N_tot = min([N_lower, N_upper])
			

		for j in range(n_bins):
			print('Working on bin {} ({:.2f}% upper points)'.format(j, fracs[j]*100))
			bin_file ='{}/stack_{}_{:.2f}%Upper.npy'.format(stack_save_dest,z,fracs[j]*100)
			
			if not have_stacks:
				# use fractions to compile a mixed mass sample for this bin
				lower_pop = math.ceil(N_tot*(1-fracs[j]))
				upper_pop = math.floor(N_tot*fracs[j])
				lower_sample = stack_lower[0:lower_pop]
				upper_sample = stack_upper[0:upper_pop]
				sample = rcfn.stack_arrays([lower_sample, upper_sample], usemask=False)
				np.save(bin_file, sample)
			elif have_stacks:
				sample = np.load(bin_file)

			sample_v = sample['v{}{}'.format(vel_type, norm)]
			#sample_v_1d = sample['v_1d_{}orm'.format(norm)]
			bootstrap_vDisp = np.zeros(1000)
			#bootstrap_vDisp_1d = np.zeros(1000)

			# find the dispersion of the cores in this bin, and associated errors, via 
			# bootstrap resampling 
			for k in range(1000):
				if(k%500 == 0): print('bootstrap resample {}/{}'.format(k, 1000))
				next_sample = np.random.choice(sample_v, size=(len(sample_v),), 
								  replace=True)
				#next_sample_1d = np.random.choice(sample_v_1d, size=(len(sample_v),), 
				#					  replace=True)
				bootstrap_vDisp[k] = stat.bDispersion(next_sample) / vDisp_all
				#bootstrap_vDisp_1d[k] = stat.bDispersion(next_sample_1d) / vDisp_all_1d

			# normalize the resultant dispersions and errors by the dispersion of the 
			# entire stacked halo
			bin_dispersions[j] = np.mean(bootstrap_vDisp)
			bin_disp_errors[j] = np.std(bootstrap_vDisp)
			#bin_dispersions_1d[j] = np.mean(bootstrap_vDisp_1d)
			#bin_disp_errors_1d[j] = np.std(bootstrap_vDisp_1d)
			print('Done. dispersion = {} +- {}'.format(bin_dispersions[j], 
								   bin_disp_errors[j]))
		
		print('saving segregation data')	
		cols = ['pcen', 'bin_vDisp', 'bin_vDisp_err']
		output = np.rec.fromarrays([fracs, bin_dispersions, bin_disp_errors], names=cols)
		#cols_1d = ['pcen', 'bin_vdisp_1d', 'bin_vDisp_err_1d']
		#output_1d = np.rec.fromarrays([fracs, bin_dispersions_1d, bin_disp_errors_1d], 
		#			       names=cols_1d)

		np.save('{}/vSegr_{}.npy'.format(data_save_dest, z), output)
		#np.save('{}/{}_vSegr_1d_{}.npy'.format(data_save_dest, normalization, 
		#					  segr_prop,z),output_1d)
		print('Done.')	


def V_vs_R(max_r = 2.5, dim=3, plot=False):
	'''
	Calculate and plot the average core velocity as a function of radius in
	stacked halos. The procedure is to bin the core data by radius, find the mean
	core velocity and uncertainty in each bin, and plot the results. 
	
	The width of the radius bins are determined such that they all hold the same
	number of data points (by taking percentiles of the core data). The number of
	bins is decided rather randomly, with the aim to hav enough bins to represent the data, 
	without having too few that a very low number of cores lands in each bin. A number
	of bins that seems to satisfy this is 2/3 the number of halos present in the working 
	stacked halo.

	:param max_r: the maximum radius (in units of r200) to cut off the analysis; going too far
		      can include far outlying cores from overlinked halos. (default = 2.5)
	:param dim: dimensions with which to preform the analysis. This simply means that if dim=3,
		    use radial velocities and 3D dispersions. If dim=1, use 1d velocities and 
		    dispersions. Valid values are 1 and 3. (default = 3)
	:param plot: whether or not to plot the results (default=False)
	:return: None	    
	'''

	if(dim != 1 and dim != 3): raise ValueError('arg \'dim\' must be 1 or 3')

	path = '/home/jphollowed/data/hacc/alphaQ/coreCatalog_merged/stackedHalos/by_redshift'
	save_dest = '/home/jphollowed/data/hacc/alphaQ/coreCatalog_merged/stackAnalysis/VvsR_cores'
	if(dim==3):fig_path = '/home/jphollowed/figs/VvsR_figs/merged_figs'
	elif(dim==1):fig_path='/home/jphollowed/figs/VvsR_1D_figs/merged_figs'
	stack_files = sorted(glob.glob('{}/stack*'.format(path)))

	for f in stack_files:
		
		stack = np.load(f)
		N_halo = int(f.split('_')[-1].split('h')[0])
		N_core = len(stack)
		z = f.split('_')[-2]
		print('\nworking on stack at z = {}'.format(z))

		if(dim == 3):
			r = stack['r_rad_norm']
			r_mask = r < max_r
			r = r[r_mask]
			v_c = stack['v_rad_coreNorm'][r_mask]
			v_d = stack['v_rad_dmNorm'][r_mask]
		if(dim == 1):
			r = stack['r_rad_2d_norm']
			r_mask = r < max_r
			r = r[r_mask]
			v_c = stack['v_1d_coreNorm'][r_mask]
			v_d = None

		n_bins = N_halo * 0.66
		bin_edges = np.percentile(r, np.linspace(0, 100, n_bins).tolist())
		avg_r = [(bin_edges[n+1] + bin_edges[n]) / 2 for n in range(len(bin_edges)-1)]
		avg_v = np.empty(len(avg_r))
		err = np.zeros(len(avg_r))

		for i in range(len(avg_v)):
			if(i%10==0): print('working on bin {}'.format(i))
			mask = (r <= bin_edges[i+1]) & (r > bin_edges[i])
			vs = v_c[mask]
			bootstrap_means = np.zeros(1000)

			for j in range(len(bootstrap_means)): 
				next_sample= np.random.choice(vs, size=len(vs), replace=True)
				bootstrap_means[j] = np.mean(next_sample)
			
			avg_v[i] = np.mean(vs)
			err[i] = np.std(bootstrap_means)
		
		cols=['avg_r', 'avg_v', 'error']
		stack_output = np.rec.fromarrays([avg_r, avg_v, err], names=cols)
		if(dim==3): np.save('{}/VvsR_cores_{}.npy'.format(save_dest, z), stack_output)
		if(dim==1): np.save('{}/VvsR_1D_cores_{}.npy'.format(save_dest, z), stack_output)
		
		if(plot):
	
			plt.rc('text', usetex=True)
			params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
			plt.rcParams.update(params)
			plt.rcParams['mathtext.fontset'] = 'custom'
			plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
			plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
			plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
			
			fig1 = plt.figure()
			ax1 = fig1.add_subplot(111)
			if(dim==3):	
				ax1.plot(r, v_c, 'x', label='radial core velocity', color=[.7, .7, 1])
				ax1.plot(avg_r, avg_v, '+r', ms=10, mew=1.5, label='bin avg velocity')
				ax1.fill_between(avg_r, avg_v-err, avg_v+err, color=[1, .7, .7])
				ax1.plot([0, 2.5], [0, 0], '--', color='black', linewidth=1.5)
				ax1.set_xlabel(r'$r/r_{200}$', fontsize=24)
				ax1.set_ylabel(r'$v_{radial}/\sigma_{v(cores),3D}$', fontsize=24)
			elif(dim==1):
				ax1.plot(r, v_c, 'x', label='projected core velocity', color=[.7, .7, 1])
				ax1.plot(avg_r, avg_v, '+r', ms=10, mew=1.5, label='bin avg velocity')
				ax1.fill_between(avg_r, avg_v-err, avg_v+err, color=[1, .7, .7])
				ax1.plot([0, 2.5], [0, 0], '--', color='black', linewidth=1.5)
				ax1.set_xlabel(r'$r_{2D}/r_{200}$', fontsize=24)
				ax1.set_ylabel(r'$v_{1D}/\sigma_{v(cores),1D}$', fontsize=24)
			ax1.set_ylim([-4, 4])
			ax1.set_xlim([0, 2.5])
			ax1.legend()

			ax1.text(0.05, 0.85, 'z = {}\n{} halos\n{} total cores'.format(z, N_halo, N_core), 
				 transform=ax1.transAxes, fontsize = 16)

			plt.savefig('{}/{}.png'.format(fig_path, z))
			print('saved figure for z = {}'.format(z))
