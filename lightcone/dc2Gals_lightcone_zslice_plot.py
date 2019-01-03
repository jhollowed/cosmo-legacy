import numpy as np
import matplotlib.pyplot as plt
import glob
import genericio as gio
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import pdb
import matplotlib as mpl

mpl.rcParams.update({'figure.autolayout': True})
params = {'text.usetex': True, 'mathtext.fontset': 'stixsans'}
mpl.rcParams.update(params)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

f = plt.figure()
ax = f.add_subplot(121, aspect='equal')
ax2 = f.add_subplot(122, aspect='equal')

dirsnap = '/gpfs/mira-fs0/projects/SkySurvey/dkorytov/projects/hdf5_to_gio/output/v4.15.0'

x = np.squeeze(gio.gio_read('{}/487.gio'.format(dirsnap), 'x'))
y = np.squeeze(gio.gio_read('{}/487.gio'.format(dirsnap), 'y'))
z = np.squeeze(gio.gio_read('{}/487.gio'.format(dirsnap), 'z'))

zMask = np.logical_and(z > 118, z < 138)

#ax.hist2d(x[zMask], y[zMask], cmap='Greys', bins = 300, norm=LogNorm())
ax.scatter(x[zMask], y[zMask], s=0.01, color='k', alpha=0.3)

dirlc = '/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/protoDC2/lightcone'
stepdirs = glob.glob('{}/lcGals*'.format(dirlc))

cmap = plt.cm.Greys
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(1, 0, cmap.N)
my_cmap = ListedColormap(my_cmap)

x = np.array([])
y = np.array([])
rs = np.array([])

maxZ = 0.2

for i in range(len(stepdirs)):
    print(stepdirs[i])
    if((1/np.linspace(1/201, 1, 500)-1)[int(stepdirs[i].split('Gals')[-1])] > maxZ): continue
    xx = np.squeeze(gio.gio_read(sorted(glob.glob('{}/*'.format(stepdirs[i])))[0], 'x'))
    yy = np.squeeze(gio.gio_read(sorted(glob.glob('{}/*'.format(stepdirs[i])))[0], 'y'))
    zz = np.squeeze(gio.gio_read(sorted(glob.glob('{}/*'.format(stepdirs[i])))[0], 'z'))
    aa = np.squeeze(gio.gio_read(sorted(glob.glob('{}/*'.format(stepdirs[i])))[0], 'a'))
    rss =  1/aa - 1
    zzMask = np.logical_and(zz > 125, zz < 130)

    x = np.append(x, xx[zzMask])
    y = np.append(y, yy[zzMask])
    rs = np.append(rs, rss[zzMask])


ax2_hist = ax2.scatter(x, y, c=rs, cmap='OrRd', s=0.01, alpha=0.3)

ax.set_xlim([0, 256])
ax.set_ylim([0, 256])
ax2.set_xlim([0, max(y)])
ax2.set_ylim([0, max(y)])
ax.set_xlabel(r'$x\>\>[\mathrm{Comv. Mpc/h}]$', fontsize=20)
ax.set_ylabel(r'$y\>\>[\mathrm{Comv. Mpc/h}]$', fontsize=20)
ax2.set_xlabel(r'$x\>\> [\mathrm{Comv. Mpc/h}]$', fontsize=20)
ax2.set_ylabel(r'$y\>\> [\mathrm{Comv. Mpc/h}]$', fontsize=20, rotation=270, labelpad=30)
plt.tick_params(axis='both', which='major', labelsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

cbar = plt.colorbar(ax2_hist, ax=ax2, orientation='horizontal')
cbar.set_label(r'$z$', fontsize=20)
cbar.ax.tick_params(labelsize=14) 
cbar.set_clim(0, 0.3)
cbar.set_alpha(1)
cbar.draw_all()

plt.show()
