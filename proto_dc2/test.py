import numpy as np
import glob
import h5py
import pdb
import pecZ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dtk

corePath = '/home/jphollowed/code/repos/cosmology/core_tracking/data/coreCatalog/'\
           'MedianVelocity/haloCores_processed.hdf5'
f = h5py.File(corePath, 'r')
step = 293
zTool = dtk.StepZ(200, 0, 500)
nHalos = len(list(f['step_{}'.format(step)].keys()))

for j in range(nHalos):
 
    halo = f['step_{}'.format(step)][list(f['step_{}'.format(step)].keys())[j]]
    center = np.array([halo.attrs['fof_halo_center_{}'.format(r)] for r in ['x', 'y', 'z']])
    x = halo['x'][:]
    y = halo['y'][:]
    z = halo['z'][:]
    vx = halo['vx'][:]
    vy = halo['vy'][:]
    vz = halo['vz'][:]

    rs = np.ones(len(vx))
    rs = rs * zTool.get_z(step)
    if(len(rs) < 50): continue

    z_pec, z_tot, v_pec, v_los, r_rel_mag, r_rel_prop, r_dist = pecZ.pecZ(x, y, z, vx, vy, vz, rs)

    pdb.set_trace()

    fig = plt.figure(1)
    fig.clf()
    ax = fig.gca(projection='3d')
    color = [['r', 'b'][rs < 0] for rs in z_pec]

    for i in range(len(z_pec)):
            ax.quiver(x[i], y[i], z[i], vx[i], vy[i], vz[i], color=color[i], length = v_pec[i]/1000, 
                      arrow_length_ratio = 0.1, pivot='tail', lw=1.3)

    ax.plot(x, y, 'o', zs=z, color='grey', mew=0)

    mx = np.median(x)
    my = np.median(y)
    mz = np.median(z)
    ax.quiver(mx, my, mz, 0-mx, 0-my, 0-mz, color='k', length=2.2, arrow_length_ratio = 0.14, 
              pivot='tail', lw=2.5)
    ax.set_xlim([np.median(x)-1.5, np.median(x)+1.5])
    ax.set_ylim([np.median(y)-1.5, np.median(y)+1.5])
    ax.set_zlim([np.median(z)-1.5, np.median(z)+1.5])

    fig.clf()
    ax1 = fig.add_subplot(111)
    bins = np.histogram(r_dist, 20)[1]
    ax1.hist(r_rel_prop, bins=bins, color=[1, .2, .2], linestyle='--', histtype='step')
    ax1.hist(r_dist, bins=bins, color='red', histtype='step')

    plt.show()
