import numpy as np
import pylab as pl
from datetime import datetime
import time
import pdb

#--------------------------------------------------------------------
from astropy.cosmology import WMAP7 as cosmo

vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2
apr = 206269.43
#--------------------------------------------------------------------
# Angular Diameter Distance (Mpc/h)
#
def Da(z):
    res = cosmo.comoving_distance(z).value*cosmo.h/(1.0+z)
    return res

def Da2(z1,z2):

    Dcz1 = (cosmo.comoving_distance(z1).value*cosmo.h)
    Dcz2 = (cosmo.comoving_distance(z2).value*cosmo.h)
    res = (Dcz2-Dcz1+1e-8)/(1.0+z2)
    return res
#--------------------------------------------------------------------
# Einstein Radius (arcsec)
#
def re_sv(sv, z1, z2):
    res = 4.0*np.pi*(sv**2.0/vc**2.0)*Da2(z1,z2)/Da(z2)*apr
    return res
#--------------------------------------------------------------------
# The function to calculate deflection angles,
# convergence, shears, and magnification
#
def lensing_signals_sie(x1, x2, lpar):
    xc1 = lpar[0]           # center of the lens, arcsec
    xc2 = lpar[1]           # center of the lens, arcsec
    q = lpar[2]             # axis ratio
    re = lpar[4]            # Einstein Radius, arcsec
    rc = lpar[3]            # core size of the lens, arcsec
    pha = lpar[5]           # position angle, degree

    phirad = np.deg2rad(pha)
    cosa = np.cos(phirad)
    sina = np.sin(phirad)

    xt1 = ((x1 - xc1) * cosa + (x2 - xc2) * sina)
    xt2 = ((x2 - xc2) * cosa - (x1 - xc1) * sina)

    phi = np.sqrt(xt2 * xt2 + xt1 * q * xt1 * q + rc * rc)
    sq = np.sqrt(1.0 - q * q)
    pd1 = phi + rc / q
    pd2 = phi + rc * q
    fx1 = sq * xt1 / pd1
    fx2 = sq * xt2 / pd2
    qs = np.sqrt(q)

    a1 = qs / sq * np.arctan(fx1)
    a2 = qs / sq * np.arctanh(fx2)

    xt11 = cosa
    xt22 = cosa
    xt12 = sina
    xt21 = -sina

    fx11 = xt11 / pd1 - xt1 * \
        (xt1 * q * q * xt11 + xt2 * xt21) / (phi * pd1 * pd1)
    fx22 = xt22 / pd2 - xt2 * \
        (xt1 * q * q * xt12 + xt2 * xt22) / (phi * pd2 * pd2)
    fx12 = xt12 / pd1 - xt1 * \
        (xt1 * q * q * xt12 + xt2 * xt22) / (phi * pd1 * pd1)
    fx21 = xt21 / pd2 - xt2 * \
        (xt1 * q * q * xt11 + xt2 * xt21) / (phi * pd2 * pd2)

    a11 = qs / (1.0 + fx1 * fx1) * fx11
    a22 = qs / (1.0 - fx2 * fx2) * fx22
    a12 = qs / (1.0 + fx1 * fx1) * fx12
    a21 = qs / (1.0 - fx2 * fx2) * fx21

    rea11 = (a11 * cosa - a21 * sina) * re
    rea22 = (a22 * cosa + a12 * sina) * re
    rea12 = (a12 * cosa - a22 * sina) * re
    rea21 = (a21 * cosa + a11 * sina) * re

    kappa = 0.5 * (rea11 + rea22)
    shear1 = 0.5 * (rea12 + rea21)
    shear2 = 0.5 * (rea11 - rea22)

    y11 = 1.0 - rea11
    y22 = 1.0 - rea22
    y12 = 0.0 - rea12
    y21 = 0.0 - rea21

    jacobian = y11 * y22 - y12 * y21
    mu = 1.0 / jacobian

    alpha1 = (a1 * cosa - a2 * sina) * re
    alpha2 = (a2 * cosa + a1 * sina) * re

    return alpha1, alpha2, kappa, shear1, shear2, mu
#--------------------------------------------------------------------
# Rotate the sources
#
def xy_rotate(x, y, xcen, ycen, phi):
	phirad = np.deg2rad(phi)
	xnew = (x - xcen) * np.cos(phirad) + (y - ycen) * np.sin(phirad)
	ynew = (y - ycen) * np.cos(phirad) - (x - xcen) * np.sin(phirad)
	return (xnew,ynew)
#--------------------------------------------------------------------
# 2D Gaussian Profile
#
def gauss_2d(x, y, par):
	(xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
	res0 = ((xnew**2)*par[4]+(ynew**2)/par[4])/np.abs(par[1])**2
	res = par[0]*np.exp(-0.5*res0)
	return res
#--------------------------------------------------------------------
# Main Function
#
def main(zl=0.1, zs=1.0, boxsize=6.0, nnn=512, sigmav=220, l_xcen=0.0, l_ycen=0.0, l_axrat=0.7, 
         l_core=0.0, l_orient=0.0, g_amp=1.0, g_sig=0.15, g_xcen=0.3, g_ycen=0.2, g_axrat=0.5, 
         g_orient=-20.00, showPlot = True):
    '''
    Preform gravitation lensing ray-tracing with lens and source objects as specified by the input 
    parameters
    
    :param zl: redshift of the lens
    :param zs: redshift of the source
    :param boxsize: the size of the field of view (arcsec)
    :param nnn: number of pixels per side
    
    :param sigmav: velocity dispersion of lens (km/s)
    :param l_xcen: x-position center of lens (arcsec)
    :param l_ycen: y-position center of lens(arcsec)
    :param l_axrat: minor-to-major axis ratio of lens
    :param l_orient: lens major-axis position angle ccw from x-axis (degrees)
    :param l_core: size of lens zore structure (arcsec)

    :param g_amp: source peak brightness value (arbitrary units)
    :param g_sig: source gaussian position scatter, i.e size (arcsec)
    :param g_xcen: x-position center of source (arcsec)
    :param g_ycen: y-position center of source (arcsec)
    :param g_axrat: minor-to-major axis ratio of source
    :param g_orient: source major-axis position angle ccw from x-axis (degrees)
    
    :param showPlot: boolean determining whether or not to display plotted result (false - save to file)

    :return: 0 if successful
    '''

    dsx = boxsize/nnn       # the size of one pixel
    xi1 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+dsx/2.0
    xi2 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+dsx/2.0
    xi1,xi2 = np.meshgrid(xi1,xi2)      # Grids on the lens plane

    #--------------------------------------------------------------------
    l_eRadius = re_sv(sigmav,zl,zs)    # Einstein Radius (arcsec)
    l_param = np.asarray([l_xcen,l_ycen,l_axrat,l_core,l_eRadius,l_orient])
    al1, al2, kap, sh1, sh2, mua =  lensing_signals_sie(xi1, xi2, l_param)
    
    #----------------------------------------------------------------------
    g_param = np.asarray([g_amp,g_sig,g_xcen,g_ycen,g_axrat,g_orient])
    g_image = gauss_2d(xi1,xi2,g_param)    # image of the source
    
    #----------------------------------------------------------------------
    yi1 = xi1-al1       # intersection of deflected light radius and the source plane
    yi2 = xi2-al2       # intersection of deflected light radius and the source plane
    gpar = np.asarray([g_amp,g_sig,g_xcen,g_ycen,g_axrat,g_orient])
    g_lensimage = gauss_2d(yi1,yi2,g_param)    # lensed images
   
    return 
    #--------------------------lens images contour------------------------
    levels = [0.15,0.30,0.45,0.60,0.75,0.9,1.05]
    fig = pl.figure(num=None,figsize=(10,5),dpi=80, facecolor='w', edgecolor='k')

    a = pl.axes([0.1,0.2,0.3,0.6])
    a.set_xlim(-boxsize/2.0,boxsize/2.0)
    a.set_ylim(-boxsize/2.0,boxsize/2.0)
    pl.xlabel("arcsec")
    pl.ylabel("arcsec")
    pl.xticks([-3,-2,-1,0,1,2,3])
    pl.yticks([-3,-2,-1,0,1,2,3])
    a.contourf(xi1,xi2,g_image,levels)
    a.contour(yi1,yi2,mua,0,colors=('g'),linewidths = 2.0)

    b = pl.axes([0.6,0.2,0.3,0.6])
    b.set_xlim(-boxsize/2.0,boxsize/2.0)
    b.set_ylim(-boxsize/2.0,boxsize/2.0)
    pl.xlabel("arcsec")
    pl.ylabel("arcsec")
    pl.xticks([-3,-2,-1,0,1,2,3])
    pl.yticks([-3,-2,-1,0,1,2,3])
    b.contourf(xi1,xi2,g_lensimage,levels)
    b.contour(xi1,xi2,mua,colors=('k'),linewidths = 2.0)

    if(showPlot):
        pl.show()
    else:
        pl.savefig('lens_{}.png'.format(str(datetime.now())))


if __name__ == '__main__':
    # run with default parameters
    
    times = np.zeros(1000)

    for i in range(len(times)):
        if(i %100 == 0): print(i)
        start = time.time()
        main(showPlot=False)
        end = time.time()
        times[i] = end - start

    time_avg = np.mean(times)
    time_err = np.std(times) / np.sqrt(len(times))
    print('Time = {} +- {}'.format(time_avg, time_err))
    pdb.set_trace()
