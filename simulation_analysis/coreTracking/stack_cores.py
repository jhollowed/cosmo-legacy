'''
Joe Hollowed
Last edited 2/7/2017

Script to stack large cluster-sized dark matter halo cores. This means to normalize each 
of the cores radius and velocities by properties of its host cluster. The following data 
saved per core in the stacked (or ensemble) halo:


- The radial position: vec{r_core} - vec{r_halo_mean} where r is a cartesian 
	position vector. This radial position is multiplied by the scale
	factor, a, given by the host halo's redshift. 

- The normalized radial position: the radial position described above, scaled by 
	the inverse sod halo radius (r200) of the host halo

- The radial velocity: ((v_core - v_halo_mean) {DOT} (r / |r| )
	where r is a cartesian position vector of the core. This radial
	velocity is then multiplied by the scale factor, a, given by
	the host halo's redshift.

- The radial "coreNorm" and "dmNorm"  velocity: the radial velocity above, 
	scaled by the inverse of the core velocity dispersion and dark matter (particle) 
	velocity dispersion of the host cluster, respectively.

- The halo-respective cartesian velocity: simply the magnitude of the relative velocity vector
	between the core and the halo center. This is denoted by v_norm (note the lack of the
	'_rad_' specifier)

- The 2d radial position: the radial position projected onto the y-z plane, so the x-axis, 
	which will be chosen as out 1-dimensional velocity, is along the "line of sight"

- The normalized 2d radial position: as described above for the 3d case

- The 1d velocity of each core:  The relative 3d velocity projected onto the x-axis 

- The "coreNorm" 1d velocity: as described for the radial case above

- Host halo tag and core tag

- Core infall mass
	
- Core infall step

- Core infall halo tag


This script makes such a stacked halo at each redshift where there is at least one halo

'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import numpy as np
import dtk
import glob
import pdb
from coreTools import unwrap_position
from astropy.cosmology import WMAP9 as cosmo
from scipy.stats import itemfreq
import numpy.lib.recfunctions as rcfn

catalog = '20f_cut'
print('\n\nReading data from {} catalog'.format(catalog))
halo_path = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/bigHalo_properties_{}.npy'
	     .format(catalog, catalog))
core_path = '/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/bigHalos'.format(catalog)
stack_path = '/home/jphollowed/data/hacc/alphaQ/coreCatalog_{}/stackedHalos/by_redshift'.format(catalog)
z_tool = dtk.StepZ(200, 0, 500)
boxL = 256

all_halos = np.load(halo_path)
halo_cores = glob.glob('{}/*_cores*.npy'.format(core_path))
halo_zs = np.unique(all_halos['z'])

for n in range(len(halo_zs)):

	# Begin stacking on next redshift (step)
	next_z = halo_zs[n]
	halos = all_halos[all_halos['z'] == next_z]
	save_dest = ('{}/stack_{:.2f}_{}halos.npy'.format(stack_path, next_z, len(halos)))
	halo_tags = []
	core_tags = []
	core_infall_steps = []
	core_infall_masses = []
	core_radii = []
	r_stack = []
	r_norm_stack = []
	v_stack = []
	v_coreNorm_stack = []
	v_dmNorm_stack = []
	v_radial_stack = []
	v_radial_coreNorm_stack = []
	v_radial_dmNorm_stack = []
	v_tangent_stack = []
	v_tangent_coreNorm_stack = []
	v_tangent_dmNorm_stack = []

	r_2d_stack = []
	r_2d_norm_stack = []
	v_1d_stack = []
	v_1d_coreNorm_stack = []

	print('\n\n{} halos at redshift {:.2f}'.format(len(halos), next_z))


	for j in range(len(halo_cores)):
		
		# collect halo info, including all cores for said halo
		cores_file = halo_cores[j]
		step = int(cores_file.split('.')[-2].split('_')[0])
		core_z = z_tool.get_z(step)
		cores = np.load(cores_file)
		tag = np.int64(cores_file.split('.')[-4])
		
		halo_idx = np.nonzero((halos['fof_halo_tag'] == tag) & (halos['z'] == core_z))
		halo = halos[halo_idx]
		a = cosmo.scale_factor(core_z) 
		
		if(len(halo) == 0): 
			# cores do not belong to a halo in this redshift slice
			continue 
		if(tag == 709670038):
			# skip weird quintuplet halo 
			continue
		if(j%10 == 0):	
			print('working on halo {}'.format(tag))
			print('{} cores'.format(len(cores)))

	#----------------------------- 3 DIMENSIONAL ------------------------------------------	
		
		# gather halo velocity/position, and core velocities/positions, 
		#  in form [vx, xy, vz]
		v_cores = np.array([cores['vx'], cores['vy'], cores['vz']])
		v_cores = np.swapaxes(v_cores, 0, 1) * a
		v_halo = np.array([halo['sod_halo_mean_{}'.format(v)] for v in ['vx','vy','vz']])
		v_halo = np.ndarray.flatten(v_halo) * a
		
		r_cores = np.array([cores['x'], cores['y'], cores['z']])
		r_cores = np.swapaxes(r_cores, 0, 1)
		r_halo = np.array([halo['sod_halo_min_pot_{}'.format(r)] for r in ['x','y','z']])
		r_halo = np.ndarray.flatten(r_halo)	

		# Adjust positions of all cores that cross the periodic box bound, 
		# to reflect accurate distances. Factor of a to all resultant positions
		# to get physical distances
		r_cores = unwrap_position(r_cores, r_halo)
		r_cores = r_cores * a
		r_halo = r_halo * a
		
		# Dot relative position with relative core velocity to get radial velocity
		r_rel = r_cores - r_halo
		r_rel_mag = np.linalg.norm(r_rel, axis=1)
		r_rel_hat = np.divide(r_rel, np.array([r_rel_mag]).T)
		v_rel = v_cores - v_halo
		v_radial = np.sum(v_rel * r_rel_hat, axis=1)
			
		# if v_radial has nans, it is beacause the relative position between the halo
		# minimum and a core is zero. In this case, any relative velocity is radial, so
		# replace nan radial velocity with magnitude of relative velocity.
		# (I've included an 'axis' argument to linalg.norm, even though there should only be
		#  one core listed in centered_cores)
		centered_cores = np.nonzero(np.isnan(v_radial))	
		v_radial[centered_cores] = np.linalg.norm(v_rel[centered_cores], axis=1)

		# Find tangential velocity. First obtain vector orthagonal to plane of motion, 
		# then cross that vector with the position vector r, normalize, and dot the result
		# with the relative velocity
		tan_dir = np.cross( np.cross(v_rel, r_rel, axisa=1), r_rel, axisa=1)
		tan_dir_mag = np.linalg.norm(tan_dir, axis=1)
		tan_dir_hat = np.divide(tan_dir, np.array([tan_dir_mag]).T)
		v_tan = np.sum(abs(v_rel * tan_dir_hat), axis=1)
		# get rid of nans, in the case of cores that have pure radial velocity
		v_tan[np.isnan(v_tan)] = 0
		
		# save the magnitudes of the relative velocities
		v_rel_mag = np.linalg.norm(v_rel, axis=1)
	
		# normalize cartesian and radial velocities and radii with respect to velocity 
		# dispersion and host halo
		r_norm = r_rel_mag / (halo['sod_halo_radius'])
		v_coreNorm = v_rel_mag / halo['core_vel_disp']
		v_dmNorm = v_rel_mag / halo['sod_halo_vel_disp']
		v_radial_coreNorm = v_radial / halo['core_vel_disp']
		v_radial_dmNorm = v_radial / halo['sod_halo_vel_disp']
		v_tan_coreNorm = v_tan / halo['core_vel_disp']
		v_tan_dmNorm = v_tan / halo['sod_halo_vel_disp']
			
		# add all radial velocities and radii to stack
		for i in range(len(cores)): halo_tags.append(halo['fof_halo_tag'][0])
		for tag in cores['core_tag']: core_tags.append(tag)
		for step in cores['infall_step']: core_infall_steps.append(step)
		for mass in cores['infall_mass']: core_infall_masses.append(mass)
		for rad in cores['radius']: core_radii.append(rad)
		for r in r_rel_mag: r_stack.append(r) 
		for r in r_norm: r_norm_stack.append(r)
		for v in v_rel_mag: v_stack.append(v)
		for v in v_coreNorm: v_coreNorm_stack.append(v)
		for v in v_dmNorm: v_dmNorm_stack.append(v)
		for v in v_radial: v_radial_stack.append(v)
		for v in v_radial_coreNorm: v_radial_coreNorm_stack.append(v)
		for v in v_radial_dmNorm: v_radial_dmNorm_stack.append(v)
		for v in v_tan: v_tangent_stack.append(v)
		for v in v_tan_coreNorm: v_tangent_coreNorm_stack.append(v)
		for v in v_tan_dmNorm: v_tangent_dmNorm_stack.append(v)

	# -----------------------1 DIMENSIONAL (2d positions) ---------------------------------------

		# repeat as above for 3d case
		v_1d = np.swapaxes(v_rel, 0, 1)[0]
		r_cores_2d = np.array([r[1:] for r in r_cores])
		r_halo_2d = r_halo[1:]
		r_rel_2d = r_cores_2d - r_halo_2d
		r_rel_mag_2d = np.linalg.norm(r_rel_2d, axis=1)
	
		r_2d_norm = r_rel_mag_2d / (halo['sod_halo_radius'])
		v_1d_coreNorm = v_1d / halo['core_vel_disp_1d']
		
		for r in r_rel_mag_2d: r_2d_stack.append(r)
		for r in r_2d_norm: r_2d_norm_stack.append(r)		
		for v in v_1d: v_1d_stack.append(v)
		for v in v_1d_coreNorm: v_1d_coreNorm_stack.append(v)
	
	halo_tags = np.array(halo_tags)
	core_tags = np.array(core_tags)
	core_infall_steps = np.array(core_infall_steps)
	core_infall_masses = np.array(core_infall_masses)
	core_radii = np.array(core_radii)
	r_stack = np.array(r_stack)
	r_norm_stack = np.array(r_norm_stack)
	v_stack = np.array(v_stack)
	v_coreNorm_stack = np.array(v_coreNorm_stack)
	v_dmNorm_stack = np.array(v_dmNorm_stack)
	v_radial_stack = np.array(v_radial_stack)
	v_radial_coreNorm_stack = np.array(v_radial_coreNorm_stack)
	v_radial_dmNorm_stack = np.array(v_radial_dmNorm_stack)
	v_tangent_stack = np.array(v_tangent_stack)
	v_tangent_coreNorm_stack = np.array(v_tangent_coreNorm_stack)
	v_tangent_dmNorm_stack = np.array(v_tangent_dmNorm_stack)
	r_2d_stack = np.array(r_2d_stack)
	r_norm_2d_stack = np.array(r_2d_norm_stack)
	v_1d_stack = np.array(v_1d_stack)	
	v_1d_coreNorm_stack = np.array(v_1d_coreNorm_stack)

	cols = ['fof_halo_tag', 'core_tag', 'infall_step', 'infall_mass', 'radius', 'v', 'v_coreNorm', 
		'v_dmNorm', 'r_rad', 'r_rad_norm', 'v_rad', 'v_rad_coreNorm', 'v_rad_dmNorm', 
		'v_tan', 'v_tan_coreNorm', 'v_tan_dmNorm', 'r_rad_2d', 'r_rad_2d_norm', 'v_1d', 
		'v_1d_coreNorm']

	stack = np.rec.fromarrays([halo_tags, core_tags, core_infall_steps, core_infall_masses, 
				   core_radii, v_stack, v_coreNorm_stack, v_dmNorm_stack, r_stack, 
				   r_norm_stack, v_radial_stack, v_radial_coreNorm_stack, 
				   v_radial_dmNorm_stack, v_tangent_stack, v_tangent_coreNorm_stack, 
				   v_tangent_dmNorm_stack, r_2d_stack, r_2d_norm_stack, v_1d_stack, 
				   v_1d_coreNorm_stack], names=cols)
	np.save(save_dest, stack)

	print('\nDone. Saved to {}\n\n'.format(save_dest))


# Make full stack across redshifts

print('Making full stack across all redshifts')
stack_files = sorted(glob.glob('{}/stack_[0-9].*.npy'.format(stack_path)))
N_halos = sum([int(s.split('_')[-1].split('h')[0]) for s in stack_files])
save_dest = '{}/stack_full_{}halos.npy'.format(stack_path, N_halos)
stacks = [np.load(s) for s in stack_files]
full_stack = np.array([], dtype=stacks[0].dtype).reshape(0, len(stacks[0]))
for stack in stacks: full_stack = rcfn.stack_arrays([full_stack, stack], usemask=False) 
np.save(save_dest, full_stack)

print('Done. Saved full stack to {}'.format(save_dest))
