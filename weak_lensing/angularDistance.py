'''
Joe Hollowed
COSMO-HEP 2017

A collection of cuntions plotting various quantities related to the planning of 
my weak-lensing mass calibration project
'''

import pdb
import numpy as np
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP7 as cosmo
from matplotlib.ticker import MultipleLocator

# =========================================================================================
# ================================== Angular Distance =====================================
# =========================================================================================

def angularDistance(L = np.array([5, 7, 10, 15, 20])):
    '''
    This function calculates and plots the lightcone cutout size in square degrees
    needed in order to subtend a square plane (in the flat sky approximation) of 
    a specific size at any distance. That is, if we want to cutout an area 10Mpc^2 centerd
    on a halo, then this code plots the necessary lightcone cutout size to achieve that
    area as a function of distance to the halo (redshift)

    :param L: numpy array of box sizes (edge length of square plane centered on halo)
    '''
    L = L.reshape(len(L), 1)

    cmap=plt.cm.plasma
    colors = cmap(np.linspace(0.1, 0.9, len(L)))
    c = cycler('color', colors)
    plt.rcParams["axes.prop_cycle"] = c

    Llabels = ['Box size = {} Mpc'.format(int(l)) for l in L]
    rMax = cosmo.comoving_distance(1).value
    thetaMin_deg = (L/rMax) * (180/np.pi)
    theta = np.array([np.linspace(thetaMin_deg[j], 90, 1000) for j in range(len(L))])
    thetaRad = theta * (np.pi/180)
    r = L /thetaRad
    MpcPerDeg = 60*(cosmo.kpc_comoving_per_arcmin(1).value/1000)
    LEdge = thetaRad * rMax
    degEdge = LEdge / MpcPerDeg
    rBad = L/(10*(np.pi/180))
    rProtoDC2 = L/(5*np.pi/180)
    pdb.set_trace()

    f = plt.figure()
    ax = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = ax.twinx()
    axZoom = plt.subplot2grid((3,1), (2,0))
    axZoom2 = axZoom.twinx()

    for j in range(len(L)):
        for a in [ax, axZoom]: a.plot(r[j], theta[j], lw=2, label=Llabels[j], color=colors[j])
        ax.plot([rBad[j], rBad[j]], [0,90], '--', lw=1.5, color=colors[j])
        #ax.fill_between([0, rProtoDC2[j]], [0,0], [90,90], color='grey', alpha=0.2)
        for a2 in [ax2, axZoom2]: a2.plot(r[j], LEdge[j], lw=0, color=colors[j])

    ax.set_xlim([10, 500])
    ax.set_ylim([0, 35])
    axZoom.set_xlim([400, rMax])
    axZoom.set_ylim([0, 2])
    ax2.set_ylim([0, 35*(np.pi/180) * rMax])
    axZoom2.set_ylim([0, 2*(np.pi/180) * rMax])
    axZoom.set_xlabel(r'$\rho\>\mathrm{(Mpc)}$', fontsize=14)

    miniTickx = MultipleLocator(25)
    miniTickx2 = MultipleLocator(100)
    miniTicky = MultipleLocator(2)
    ax.xaxis.set_minor_locator(miniTickx)
    axZoom.xaxis.set_minor_locator(miniTickx2)
    for a in [ax, axZoom]: 
        a.yaxis.set_minor_locator(miniTicky)
        a.grid(True, linestyle='--')
        a.xaxis.grid(b=True, which='minor', linestyle='-', alpha=0.2)
        a.set_ylabel(r'$\theta\>\mathrm{(deg)}$', fontsize=14)
    for a2 in [ax2, axZoom2]: 
        a2.yaxis.grid(b=True, which='minor', linestyle='-', alpha=0.2)
        a2.set_ylabel(r'$L\>\mathrm{(Mpc)}$', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.show()

# =========================================================================================
# ================================== Redshift Distribution ================================
# =========================================================================================

def changNz(fiducial_only = False):
    '''
    Plots the n(z) fit from Chang+2013 for both n_raw (raw galaxy counts per arcmin) and 
    n_eff (effective lensing galaxy counts per arcmin) for the optimistic, fiducial, and 
    conservative cases (recreation of fig.7 from Chang+2013)
    
    :param fiducial_only: whether or not to plot only the fiducial distribution. In 
                          this case, three different distributions are shown with different
                          linestyles; the raw n_eff, raw+blending corrections, and
                          raw + blending + masking corrections
    '''
    
    if(fiducial_only):
        labels = [r'raw $n_\mathrm{eff}$', r'   + blending ($d$=2 arcsec)', 
                  r'   + masking (15%)', r'uncut $n$']
        linestyles = ['--', '-', ':']
    else:
        labels = [r'optimistic $n_\mathrm{eff}$', r'fiducial $n_\mathrm{eff}$', 
                  r'conservative $n_\mathrm{eff}$', r'uncut $n$']

    neff_fiducial = np.array([37, 31, 26])
    alpha = np.array([[1.23], [1.24], [1.28], [1.25]])
    z0 = np.array([[0.59], [0.51], [0.41], [1.0]])
    beta = np.array([[1.05], [1.01], [0.97], [1.26]])
    z_m = np.array([[0.89], [0.83], [0.71], [1.22]])
    z = np.linspace(0, 4, 500)

    cmap=plt.cm.plasma
    colors = cmap(np.linspace(0.1, 0.9, 3))
    c = cycler('color', colors)
    plt.rcParams["axes.prop_cycle"] = c
    middle_color = c.by_key()['color'][1]

    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    fit = (z**alpha) * np.exp(-((z/z0)**beta))
    rawNorm = max(fit[1]) / max(fit[-1])
    fit[-1] = fit[-1] * rawNorm * neff_fiducial[1]
    
    if(fiducial_only):
        for j in range(len(neff_fiducial)):
            adjust_fit = fit[1] * neff_fiducial[j]
            ax.plot(z, adjust_fit, lw=2, linestyle=linestyles[j], color=middle_color,
                    label=labels[j])
    else:
        adjust_fit = fit[:-1] * neff_fiducial[1]
        for k in range(3):
            ax.plot(z, adjust_fit[k], lw=2, label=labels[k])

    ax.plot(z, fit[-1], '-.k', lw=1.3, label=labels[-1], alpha=0.7)
    ax.legend(fontsize=12)
    ax.set_xlabel(r'$z$', fontsize=14)
    ax.set_ylabel(r'$n_\mathrm{eff}\>\>(\mathrm{arcmin}^{-2})$', fontsize=14)
    ax.grid(linestyle='--')
    ax.set_aspect(0.36)
    plt.tight_layout()
    plt.show()
