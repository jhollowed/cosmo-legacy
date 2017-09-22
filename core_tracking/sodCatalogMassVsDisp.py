'''
Joe Hollowed
COSMO-HEP 2017

This script gathers and plots mass vs. velocity-dispersion data from the AlphaQ SO catalog, 
including a comparison to the Evrard+ 2003 best fit
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import pdb
import glob
import numpy as np
import dtk
from dtk import gio as gio
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP7 as cosmo

p = '/media/luna1/dkorytov/data/AlphaQ/sod/'
f = np.array(glob.glob('{}*s'.format(p)))
steps = np.array([int(ff.split('.')[-2].split('-')[-1]) for ff in f])
zTool = dtk.StepZ(200, 0, 500)

sortMask = np.argsort(steps)
steps = steps[sortMask][len(steps)-26:]
f = f[sortMask][len(f)-26:]
i=-1

for step in steps:
    i+=1
    idx = np.where(steps == step)
    z = zTool.get_z(step)
    h = cosmo.H(z).value /100
    a = 1/(1+z)
    ef = cosmo.efunc(z)
    sod = f[idx][0]

    tag = gio.gio_read(sod, 'fof_halo_tag')
    mass = gio.gio_read(sod, 'sod_halo_mass')
    disp = gio.gio_read(sod, 'sod_halo_vel_disp')

    fitMass = np.linspace(3e13, 1e16, 100)
    fit = lambda sigDM15,alpha: sigDM15 * ((fitMass) / 1e15)**alpha
    fitDisp = fit(1082.9, 0.3361)
    fitDisp_err1 = fit(1082.9 + 4, 0.3361 + 0.0026)
    fitDisp_err2 = fit(1082.9 + 4, 0.3361 - 0.0026)
    fitDisp_err3 = fit(1082.9 - 4, 0.3361 + 0.0026)
    fitDisp_err4 = fit(1082.9 - 4, 0.3361 - 0.0026)
    
    
    fig = plt.figure(0)
    fig.clf()
    plt.plot(mass[mass*ef > 1e14]*ef, disp[mass*ef>1e14]*a, 'xr')
    plt.plot(fitMass, fitDisp, '--k', lw=2)
    plt.plot(fitMass, fitDisp_err1, '--k', lw=0.8)
    plt.plot(fitMass, fitDisp_err2, '--k', lw=0.8)
    plt.plot(fitMass, fitDisp_err3, '--k', lw=0.8)
    plt.plot(fitMass, fitDisp_err4, '--k', lw=0.8)
    plt.yscale('log')
    plt.xscale('log')

    plt.ylim([300, 1400])
    plt.xlim([7e13, 2e15])


    plt.title('step {}'.format(step))
    plt.grid()
    fig.savefig('alphaq_figs/{}.png'.format(chr(ord('a')+i)))
    print('step {}'.format(step))
