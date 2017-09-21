import dtk
import glob
import numpy as np
import matplotlib.pyplot as plt
import pdb
from astropy.constants import M_sun
from astropy.cosmology import WMAP7 as cosmo
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as plticker

print('Loading data')
path = '/media/luna1/dkorytov/data/AlphaQ/sod'
f = glob.glob('{}/*sodproperties'.format(path))
z_step = dtk.StepZ(200, 0, 500)
print('finding snapshots z\'s')
steps = [int(dat.split('.')[-2].split('-')[-1]) for dat in f]
z_cat = [z_step.get_z(step) for step in steps]
print('masking data to redshift range 0-1.5')
z_mask = np.array(z_cat) <= 1.5
f = np.array(f)[z_mask]
steps = np.array(steps)[z_mask]
z_cat = np.array(z_cat)[z_mask]

m_cat = [dtk.gio_read(r, 'sod_halo_mass').tolist() for r in f]
d_cat = [dtk.gio_read(r, 'sod_halo_vel_disp').tolist() for r in f]

print('gathering mass and dispersion data for masked halos')
m = []
d = []
z = []
for n in range(len(m_cat)):
	halo_m = m_cat[n]
	halo_d = d_cat[n]
	for i in range(len(halo_m)):
		m.append(halo_m[i])
		d.append(halo_d[i])
		z.append(z_cat[n])

print('normalizing mass by h(z)')
m = np.array(m)
d = np.array(d)
z = np.array(z)
mask_guess = m >= 1e13
m = m[mask_guess]
d = d[mask_guess]
z = z[mask_guess]
h = np.array([1/(cosmo.H(zi).value/100) for zi in z])
mask = (m) >= 1e14

print('masking data to fit mass range >=1e14')
m_big = m[mask] * h[mask]
z_big = z[mask]
d_big = d[mask] * (1/(1+z[mask]))

pdb.set_trace()

t_x = np.linspace(7e13, 2e15, 300)
sig_dm15 = 1082.9
alpha = 0.3361
t_y = sig_dm15 * (( (t_x) / (1e15) )**alpha)


# preforming Least Squares on AlphaQuadrant Data
# (fitting to log linear form, as in Evrard et al.)
# X = feature matrix (masses)
# P = parameter matrix (sig_dm15(log intercept) and alpha(log slope))
#pdb.set_trace()
#X = np.array([np.log(mi / 1e15) for mi in m_big])
#X = np.vstack([X, np.ones(len(X))]).T
#P = np.linalg.lstsq(X, np.log(d_big))[0]
#alpha_fit = P[0]
#sig_dm15_fit = np.e**P[1]
#fit_x = np.linspace(7e13, 2e15, 300)
#fit_y = sig_dm15_fit * (( (fit_x) / (1e15) )**alpha_fit)

# plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
p = ax.loglog(m_big, d_big, 'rx', markersize=8)

t = ax.loglog(t_x, t_y, '--b', linewidth = 1.2)
#fit = ax.loglog(fit_x, fit_y, 'b', linewidth = 1.2)

ax.set_ylim([300, 1400])
ax.set_xlim([7e13, 2e15])

#ax.yaxis.set_major_formatter(ScalarFormatter())
#loc = plticker.MultipleLocator(base=100)
#ax.yaxis.set_major_locator(loc)
plt.show()

