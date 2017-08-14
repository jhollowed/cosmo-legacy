'''
Joe Hollowed
HEP 5/2017

This script plots cores in velocity-position phase space, along with a third (color) axis of 
infall time. 
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dtk
import glob
import coreTools
import pdb	

def makeDiagram(r, v, i, hist=False, bin_i= True):
	'''
	This function creates a phase-space diagram of clustocentric radial distance vs. clustocentric
	velocity, with a color axis indicating infall time.

	:param r: array-like of normalized core clustocentric radial positions
	:param v: array-like of normalized core clustocentric 3D velocity magnitudes
	:param i: array-like of core infall times (in Gyr)
	:param hist: whether or not to plot the data as a 2d histogram (scatter points by default)
	:praram bin_i: whether or not to gather all cores into subpopulations of infall
				 time. If True, subpopulations are created to have equal statistics, 
				 so 25% percentiles are used to assign membership. If False, treat 
				 infall times continuously. 
	'''

	# bin cores by infall time, if being used
	if(bin_i == True): 
		i = coreTools.equal_bins(i, n_bins=4)
		colorMap = ['b', 'g', 'gold', 'r']
	else:
		colorMap = 'jet'
		
	fig = plt.figure()
	ax = fig.add_subplot(111)	
	ax.scatter(r, v, c=i, s=50, lw=0, alpha=0.7, cmap = colorMap)
	for axis in ['top','bottom','left','right']:
  		ax.spines[axis].set_linewidth(1.5)
	ax.set_xlabel('R_3d / R_vir', fontsize=18)
	ax.set_ylabel('V_3d / vDisp_3d')

	plt.show()
	
def make_subPops(cat, n, cosmo):
	'''
	Gather trajectory information for every core of all n largest halos at z=0
	
	:param cat: name of catalog to grab cores from
	:param n: max number of halos to gather cores from (n largest halos in sim)
	:param cosmo: an astropy cosmology instance
	'''

	# Get data, find n largest halos in sim volume
	halo_path = '/media/luna1/dkorytov/data/AlphaQ/core_catalog4_20f'
	halos = np.load('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/'
			'bigHalo_properties_{}.npy'.format(cat, cat))
	n_mask = np.argsort(halos['sod_halo_mass'])[0:n]
	halos = halos[n_mask]
	big_halo_tags = halos['fof_halo_tag']
	halo_cores = glob.glob('{}/*499*.coreproperties'.format(halo_path))[0]

	# Filter out all cores belonging to halos that were too small to make the mass cut
	# in building bigHalo_porperties (>10^14)
	pdb.set_trace()
	core_parents = np.array([coreTools.mask_tag(tag) for tag in 
				 dtk.gio_read(halo_cores, 'fof_halo_tag')])
	cluster_mask = np.array([tag in big_halo_tags for tag in core_parents])
	core_parents = core_parents[cluster_mask]

	# Also filter out all cores but those belonging to the biggest halo in the sim volume 
	# henceforth known as fatty)
	fatty = halo_prop[np.argmax(halo_prop['sod_halo_mass'])]
	fatty_id = fatty['fof_halo_tag']
	fatty_mask = np.array([fatty_id==tag for tag in core_parents])

	# Process cores
	halo_cores_mask = coreTools.process_cores(halo_cores)
	fatty_cores_mask = coreTools.process_cores(fatty_cores)
	
	# find clustocentric positions and velocities for each core of the fatty
	halo_core_relPos = coreTools.core_locations(fatty_cores, fatty, 0)
	r = halo_core_relPos['r_rel_mag']
	halo_core_relVel = coreTools.velocity_components(fatty_cores, fatty, 0)
	v = halo_core_relVel['v_rel_mag']
	i = coreTools.step_to_Gyr(gio.gio_read(fatty_core,'infall_step'), cosmo)

	makeDiagram(r/fatty['sod_halo_radius'], v/fatty['sod_halo_vel_disp'], i)	
