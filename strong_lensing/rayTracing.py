'''
Nan Li
Joe Hollowed
COSMO-HEP 2017

This module contains a collection of functions for preforming ray-tracing of a gravitationally-lensed 
system, supporting arbitrary lens and source morphology. The main function is doRayTrace()
'''

import time
import pdb
import pylab as pl
import numpy as np
from datetime import datetime
import astropy.constants as const
from astropy.cosmology import WMAP7 as cosmo

#--------------------------------------------------------------------

#vc = 2.998e5 #km/s
vc = const.c.value / 1000 # km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2
apr = 206269.43

#--------------------------------------------------------------------

# Angular Diameter Distance (Mpc/h)
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
def re_sv(sv, z1, z2):
    res = 4.0*np.pi*(sv**2.0/vc**2.0)*Da2(z1,z2)/Da(z2)*apr
    return res

#--------------------------------------------------------------------

# The function to calculate deflection angles,
# convergence, shears, and magnification
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
def xy_rotate(x, y, xcen, ycen, phi):
	phirad = np.deg2rad(phi)
	xnew = (x - xcen) * np.cos(phirad) + (y - ycen) * np.sin(phirad)
	ynew = (y - ycen) * np.cos(phirad) - (x - xcen) * np.sin(phirad)
	return (xnew,ynew)

#--------------------------------------------------------------------

# 2D Gaussian Profile
def gauss_2d(x, y, par):
	(xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
	res0 = ((xnew**2)*par[4]+(ynew**2)/par[4])/np.abs(par[1])**2
	res = par[0]*np.exp(-0.5*res0)
	return res

#---------------------------------------------------------------------

def doRayTrace(zl=0.1, zs=1.0, boxsize=6.0, nnn=512, sigmav=220, l_xcen=0.0, l_ycen=0.0, l_axrat=0.7, 
               l_core=0.0, l_orient=0.0, g_amp=1.0, g_sig=0.15, g_xcen=0.3, g_ycen=0.2, g_axrat=0.5, 
               g_orient=-20.00, plot = True, showPlot = True):
    '''
    Preform gravitation lensing ray-tracing with lens and source objects as specified by the input 
    parameters
    
    :param zl: redshift of the lens
    :param zs: redshift of the source
    :param boxsize: the side length of the field of view (arcsec)
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
    
    :param plot: boolean whether or not to plot the rayTracing result (if True, call plotLens())
    :param showPlot: boolean determining whether or not to display plotted result (False = save to file)

    :return: 0 if lens is plotted, else return lens/source information
    '''

    #---------------------------------------------------------------------
    
    # Build lens-plane grid
    pixelSize = boxsize/nnn
    lensGrid_1d = np.linspace(-boxsize/2.0, boxsize/2.0-pixelSize, nnn) + pixelSize/2.0
    lensGrid_x, lensGrid_y = np.meshgrid(lensGrid_1d, lensGrid_1d)

    l_eRadius = re_sv(sigmav, zl, zs)    # Einstein Radius (arcsec)
    l_param = np.asarray([l_xcen, l_ycen, l_axrat, l_core, l_eRadius, l_orient])
    al1, al2, kap, sh1, sh2, mua =  lensing_signals_sie(lensGrid_x, lensGrid_y, l_param)
    
    # Build source-plane grid 
    # (intersection of deflected light radius and the source plane)
    sourceGrid_x = lensGrid_x - al1
    sourceGrid_y = lensGrid_y - al2
    
    # Make images
    g_param = np.asarray([g_amp, g_sig, g_xcen, g_ycen, g_axrat, g_orient])
    g_sourceImage = gauss_2d(lensGrid_x, lensGrid_y, g_param)
    g_lensImage = gauss_2d(sourceGrid_x, sourceGrid_y, g_param)
    
    if(plot):
        plotLens(boxsize, lensGrid_x, lensGrid_y, sourceGrid_x, sourceGrid_y, 
                 g_sourceImage, g_lensImage, mua, showPlot)

    return [al1, al2, kap, sh1, sh2, mua, g_sourceImage, g_lensImage]

#---------------------------------------------------------------------
   
def plotLens(boxsize, lensGrid_x, lensGrid_y, sourceGrid_x, sourceGrid_y, g_sourceImage, 
             g_lensImage, mua, showPlot = True):
    '''
    plot the lens and source planes as images - the source is a filled contour plot
    and the lens radius is shown as a solid curve

    :param boxsize: the side-length of the field of view (arcsec)
    :param lensGrid_x:
    :param lensGrid_y:
    :param sourceGrid_x:
    :param sourceGrid_y:
    :param g_sourceImage:
    :param g_lensImage:
    :param mua:
    :param showPlot:
    :return: None, display or save figure
    ''' 
     
    levels = [0.15,0.30,0.45,0.60,0.75,0.9,1.05]
    fig = pl.figure(num=None,figsize=(10,5),dpi=80, facecolor='w', edgecolor='k')

    a = pl.axes([0.1,0.2,0.3,0.6])
    a.set_xlim(-boxsize/2.0, boxsize/2.0)
    a.set_ylim(-boxsize/2.0, boxsize/2.0)
    pl.xlabel("arcsec")
    pl.ylabel("arcsec")
    pl.xticks([-3,-2,-1,0,1,2,3])
    pl.yticks([-3,-2,-1,0,1,2,3])
    a.contourf(lensGrid_x, lensGrid_y, g_sourceImage, levels)
    a.contour(sourceGrid_x, sourceGrid_y, mua, 0, colors=('g'), lw = 2.0)

    b = pl.axes([0.6,0.2,0.3,0.6])
    b.set_xlim(-boxsize/2.0, boxsize/2.0)
    b.set_ylim(-boxsize/2.0, boxsize/2.0)
    pl.xlabel("arcsec")
    pl.ylabel("arcsec")
    pl.xticks([-3,-2,-1,0,1,2,3])
    pl.yticks([-3,-2,-1,0,1,2,3])
    b.contourf(lensGrid_x, lensGrid_y, g_lensImage, levels)
    b.contour(lensGrid_x, lensGrid_y, mua, colors=('k'), lw = 2.0)

    if(showPlot):
        pl.show()
    else:
        pl.savefig('lens_{}.png'.format(str(datetime.now())))
    return 0

#---------------------------------------------------------------------

def timeTracing(runs = 1000):
    '''
    Function to get the time+-err of the ray-tracing functions by averaging over multiple runs
    
    :param runs: number of ray-tracing computaitons to average over
    :return: the mean execution time, and assocaited SDOM, of all runs
    '''
    
    np.random.seed(101)
    orient = np.random.random(size = runs) * 360
    times = np.zeros(runs)

    for i in range(len(times)):
        if(i %100 == 0): 
            print('lens {}/{}'.format(i, runs))
        start = time.time()
        doRayTrace(g_orient = orient[i], showPlot=False)
        end = time.time()
        times[i] = end - start

    time_avg = np.mean(times)
    time_err = np.std(times) / np.sqrt(len(times))
    return [time_avg, time_err] 

#---------------------------------------------------------------------

if __name__ == '__main__':
    '''
    provided to quickly run with default parameters
    '''
    doRayTrace() 
