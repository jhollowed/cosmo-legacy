'''
Joe Hollowed
COSMO-HEP 2017
Last edited 7/19/17
'''

import numpy as np
import matplotlib.pyplot as plt
import pdb
from astropy.constants import M_sun
from astropy.cosmology import WMAP7 as cosmo
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as plticker
import matplotlib.colors as colors
import h5py
import dtk

def plot_sigVm(catalog=0):

	catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
	fig_path = '/home/jphollowed/figs/dispVmass_figs'
	halo_path = ('/home/jphollowed/data/hacc/alphaQ/coreCatalog/{}/halo_cores.hdf5'
		     .format(catalogName))
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
	core_counts = []

	for j in range(len(zs)):
		z = zs[j]
		a = cosmo.scale_factor(z)
		h = cosmo.H(z).value/100
		thisStep = alphaQ[steps[j]]
		halo_tags = thisStep.keys()
		halos = [thisStep[tag].attrs for tag in halo_tags]
		
		halo_masses += [halo['sod_halo_mass']*h for halo in halos]
		halo_vDisp += [halo['sod_halo_vel_disp']*a for halo in halos]
		core_vDisp += [halo['sod_halo_core_vel_disp']*a for halo in halos]
		core_counts += [thisStep[tag]['core_tag'].size for tag in halo_tags]

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

	# plotting
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	p = ax.loglog(halo_masses, halo_vDisp, '^', markersize=8, color='black', zorder=1, 
		      label='DM dispersion')
	ax.hold(True)
	p_cores = ax.scatter(halo_masses, core_vDisp, lw=0, c=core_counts, cmap='PuBu', 
			     zorder=2, norm=colors.LogNorm(vmin=30, vmax=200), 
			     label='Core dispersion')
	cbar=plt.colorbar(p_cores, ticks=np.linspace(0, 200, 21), extend='max')
	cbar.ax.set_yticklabels([str(int(i)) for i in np.linspace(0, 200, 21)])

	t = ax.loglog(t_x, t_y, '--b', linewidth = 1.2)
	t_err = plt.fill_between(t_x, t_y_low, t_y_high, color=[0.7, 0.7, 1])
	fit = ax.loglog(fit_x, fit_y, 'b', linewidth = 1.2)
	fit_cores = ax.loglog(fit_x, fit_y_cores, 'g', linewidth = 1.2)

	ax.set_ylim([300, 1550])
	ax.set_xlim([4.5e13, 2e15])
	
	ax.set_ylabel('vel_disp', fontsize=16)
	ax.set_xlabel('h(z)M_{200}', fontsize=16)
	ax.legend(loc=4)
	ax.yaxis.set_major_formatter(ScalarFormatter())
	loc = plticker.MultipleLocator(base=100)
	ax.yaxis.set_major_locator(loc)
	#plt.text(0.05, 0.8, 'Dm fit: sig_dm15 = {:.2f}, alpha = {:.2f}'
	#		      .format(sig_dm15_fit, alpha_fit),
	#	 transform=ax.transAxes, fontsize = 14)
	#plt.text(0.05, 0.75,'Core fit: sig_dm15 = {:.2f}, alpha = {:.2f}'
	#		      .format(sig_dm15_fit_cores, 
	#	 alpha_fit_cores),transform=ax.transAxes, fontsize = 14)
	#plt.text(0.05, 0.7, 'Bias = sig_cors / sig_DM = {:.2f}'
	#	 .format(sig_dm15_fit_cores/sig_dm15_fit),
	#	 transform=ax.transAxes, fontsize = 14)
	plt.text(0.05, 0.8, '{}'.format(catalogName),transform=ax.transAxes, fontsize = 14)
		
	#plt.savefig('{}/{}_members.png'.format(fig_path, minN))
	plt.show()

