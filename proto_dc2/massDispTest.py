'''
Joe Hollowed
COSMO-HEP 2017
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import pdb
import h5py
import numpy as np
import pylab as plt
import clstrTools as ct
import dispersionStats as stat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from astropy.cosmology import WMAP7 as cosmo
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

def makePlot(sigma, sigmadm, realMass):
    '''
    Plot the mass-vDisp relation for each halo mass passed, using both the galaxy and particle-based
    velocity dispersions. This function is meant to be called by findMasses() below.

    :param sigma: An array of galaxy-based dispersions
    :param sigmadm: An array of particle-based dispersions
    :param realMass: the SO mass of the halo
    '''

    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)
    
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(realMass, sigmadm, 'xr', ms=6, mew=1.6, alpha=0.5, label='particle-based dispersions')
    ax.plot(realMass, sigma, '.b', ms=6, alpha=0.5, label='galaxy-based dispersions')

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


    ax.set_ylabel(r'$\sigma_v$ (km/s)', fontsize=20)
    ax.set_xlabel(r'$m_{200}$ (M$_{sun} h^{-1}$)', fontsize=20)
    ax.legend(loc='upper left', fontsize=12)

    plt.show()


def findMasses():
    '''
    This function was written as a validation test for the protoDC2 catalog - it uses 
    the observed galaxy redshifts that I have created in the catalog to measure galaxy-based
    velocity-dispersions for cluster-sized halos, and compares the results to the particle-based
    SOD velocity-dispersions. If the galaxies are placed correctly, and the observed redshifts are 
    constructed correctly, then we should see the vDisp-Mass relation roughly match, whether we are 
    using galaxy or particle based dispersions. This rough match should include a systematic bias in
    the galaxy-based dispersions. This function is meant to call makePlots() above.
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
        pdb.set_trace()

        z = halo['redshiftObserver']
        zHost = halo.attrs['halo_z']
        zHost_err = halo.attrs['halo_z_err']
        aHost = 1/(1+zHost)
        ez[j] = cosmo.efunc(zHost)
        
        realMass[j] = halo.attrs['sod_halo_mass'] * ez[j]

        sodDisp[j] = halo.attrs['sod_halo_vel_disp'] * aHost

        pecV = ct.LOS_properVelocity(z, zHost)
        galDisp[j]  = stat.bDispersion(pecV) * aHost
 
    makePlot(galDisp, sodDisp, realMass)

if(__name__ == '__main__'):
    findMasses()
