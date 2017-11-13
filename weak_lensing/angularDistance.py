import pdb
import numpy as np
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP7 as cosmo
from matplotlib.ticker import MultipleLocator

L = np.array([5, 7, 10, 15, 20])
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
