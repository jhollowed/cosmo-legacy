# Joe Hollowed
# CPAC 2018

# This module contains a collection of functions for performing
# sanity checks on the new lightcone interpolation data. 

import numpy as np
from dtk import gio
import matplotlib.pyplot as plt
import pdb

dataPath = '/homes/jphollowed/data/hacc/alphaQ/lc_output_downs'
interp_data = '{}/lc_intrp_output_d.432'.format(dataPath)
extrap_data = '{}/lc_intrp_output_d.432'.format(dataPath)

print('Reading interpolation data')
ix = gio.gio_read(interp_data, 'x')
iy = gio.gio_read(interp_data, 'y')
iz = gio.gio_read(interp_data, 'z')
ivx = gio.gio_read(interp_data, 'vx')
ivy = gio.gio_read(interp_data, 'vy')
ivz = gio.gio_read(interp_data, 'vz')
iid = gio.gio_read(interp_data, 'id')
ia = gio.gio_read(interp_data, 'a')


print('Reading extrapolation data')
ex = gio.gio_read(extrap_data, 'x')
ey = gio.gio_read(extrap_data, 'y')
ez = gio.gio_read(extrap_data, 'z')
evx = gio.gio_read(extrap_data, 'vx')
evy = gio.gio_read(extrap_data, 'vy')
evz = gio.gio_read(extrap_data, 'vz')
eid = gio.gio_read(extrap_data, 'id')
ea = gio.gio_read(extrap_data, 'a')

print('binning data')

hist_ix = np.histogram(ix, bins=2000)
hist_ex = np.histogram(ex, bins=hist_ix[1])
hist_iy = np.histogram(iy, bins=2000)
hist_ey = np.histogram(ey, bins=hist_iy[1])
hist_iz = np.histogram(iz, bins=2000)
hist_ez = np.histogram(ez, bins=hist_iz[1])

hist_ivx = np.histogram(ivx, bins=2000)
hist_evx = np.histogram(evx, bins=hist_ivx[1])
hist_ivy = np.histogram(ivy, bins=2000)
hist_evy = np.histogram(evy, bins=hist_ivy[1])
hist_ivz = np.histogram(ivz, bins=2000)
hist_evz = np.histogram(evz, bins=hist_ivz[1])

binx = (hist_ix[1] + np.diff(hist_ix[1])[0]/2)[:-1]
biny = (hist_iy[1] + np.diff(hist_iy[1])[0]/2)[:-1]
binz = (hist_iz[1] + np.diff(hist_iz[1])[0]/2)[:-1]
binvx = (hist_ivx[1] + np.diff(hist_ivx[1])[0]/2)[:-1]
binvy = (hist_ivy[1] + np.diff(hist_ivy[1])[0]/2)[:-1]
binvz = (hist_ivz[1] + np.diff(hist_ivz[1])[0]/2)[:-1]

np.savez('lc_hists.npz', ix=hist_ix[0], ex=hist_ex[0], binx=binx,
			 iy=hist_iy[0], ey=hist_ey[0], biny=biny,
			 iz=hist_iz[0], ez=hist_ez[0], binz=binz,
			 ivx=hist_ivx[0], evx=hist_evx[0], binvx=binvx,
			 ivy=hist_ivy[0], evy=hist_evy[0], binvy=binvy,
			 ivz=hist_ivz[0], evz=hist_evz[0], binvz=binvz)
pdb.set_trace()
