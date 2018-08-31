import struct
import glob
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import seaborn as ss

def downsample(arr, factor=0.01):
    return np.random.choice(arr, int(len(arr)*factor), replace=False)

def makePlot(outDir, bins=50, showBeams=False, downsFrac = 0.10, cm='plasma'):

    mpl.rcParams.update({'figure.autolayout': True})
    params = {'text.usetex': True, 'mathtext.fontset': 'stixsans'}
    mpl.rcParams.update(params)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    
    subdirs = glob.glob("{}/*".format(outDir))
    
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue

    steps = [int(s.split('Cutout')[-1]) for s in subdirs]
    steps = sorted(steps)
    dt = np.dtype('<f')
    x = np.array([])
    y = np.array([])
    z = np.array([])
    rs = np.array([])
    theta = np.array([])
    phi = np.array([])
    thetaRot = np.array([])
    phiRot = np.array([])
    pid = np.array([], dtype=int)

    conv = (np.pi / 180)
    arcsec = 3600

    f = plt.figure(0)
    if(showBeams):
        ax1 = plt.subplot2grid((3,2), (0,0), colspan=2)
        ax2 = plt.subplot2grid((3,2), (1,0), rowspan=2)
        ax3 = plt.subplot2grid((3,2), (1,1), rowspan=2)
    else:
        ax2 = f.add_subplot(121)
        ax3 = f.add_subplot(122)
    
    for i in range(len(steps)):
        
        step = steps[i]
        xf = "{0}/{1}{2}/x.{2}.bin".format(outDir, prefix, step)
        yf = "{0}/{1}{2}/y.{2}.bin".format(outDir, prefix, step)
        zf = "{0}/{1}{2}/z.{2}.bin".format(outDir, prefix, step)
        rsf = "{0}/{1}{2}/redshift.{2}.bin".format(outDir, prefix, step)
        tf = "{0}/{1}{2}/theta.{2}.bin".format(outDir, prefix, step)
        pf = "{0}/{1}{2}/phi.{2}.bin".format(outDir, prefix, step)
        idf = "{0}/{1}{2}/id.{2}.bin".format(outDir, prefix, step)

        if(len(np.fromfile(xf, dtype=dt)) == 0):
            continue
        
        downs = downsample(np.arange(len(np.fromfile(xf, dtype=dt))), downsFrac)
        x = np.hstack([x, np.fromfile(xf, dtype=dt)[downs]])
        y = np.hstack([y, np.fromfile(yf, dtype=dt)[downs]])
        z = np.hstack([z, np.fromfile(zf, dtype=dt)[downs]])
        rs = np.hstack([rs, np.fromfile(rsf, dtype=dt)[downs]])
        thetaRot = np.hstack([thetaRot, np.fromfile(tf, dtype=dt)[downs] / arcsec])
        phiRot = np.hstack([phiRot, np.fromfile(pf, dtype=dt)[downs] / arcsec])
        pid = np.hstack([pid, np.fromfile(idf, dtype=int)[downs]])

    # We only have the raw (unrotated) carteisan comoving positions, and the 
    # cluster-centric (rotated) angular coordinates, so for the full plot, we need
    # the rotated 3d positions, and the raw angular coordinates

    # Get rotated positions
    r = np.sqrt(x**2 + y**2 + z**2)
    xRot = r*np.sin(thetaRot*conv)*np.cos(phiRot*conv)
    yRot = r*np.sin(thetaRot*conv)*np.sin(phiRot*conv)
    zRot = r*np.cos(thetaRot*conv)

    # Get raw angular coords 
    theta = np.hstack([theta, np.arccos(z/r) * 1/conv])
    phi = np.hstack([phi, np.arctan(y/x) * 1/conv])
    
    thetaSpan = np.max(theta) - np.min(theta) 
    phiSpan = np.max(phi) - np.min(phi)
    thetaSpanRot = np.max(thetaRot) - np.min(thetaRot) 
    phiSpanRot = np.max(phiRot) - np.min(phiRot)
   
    if(showBeams):
        #s1 = ax1.scatter(x, y, z, c=rs, s=2, cmap='plasma')
        cmap_bg = mpl.cm.get_cmap('plasma')
        cmap_bg = cmap_bg(0.0)
        ax1.hist2d(rs, yRot, bins=[100,int(100/6)], cmap='plasma', norm=mpl.colors.LogNorm(vmin=10))
        #ax1.set_facecolor(cmap_bg)

        a = np.linspace(1/201, 1, 500)
        zall =1/a-1
        zsteps = zall[steps]
        for sss in zsteps:
            ax1.plot([sss, sss], [np.min(yRot), np.max(yRot)], '-', color='white', lw=1)

        #f.colorbar(s1)
        ax1.set_xlabel(r'$z$', fontsize=14)
        ax1.set_ylabel(r'$y\>\>\mathrm{[Mpc/h]}$', fontsize=14)
        #ax1.set_zlabel('z')
    
    pdb.set_trace()
    ax2.hist2d(phi, theta, bins, norm=mpl.colors.LogNorm(), cmap=cm)
    ax3.hist2d(phiRot, thetaRot, bins, norm=mpl.colors.LogNorm(), cmap=cm)
    ax2.set_xlabel(r'$\boldsymbol{{\phi}}\>\>\mathrm{{[deg]}}$'+'\n'+\
                   r'$\mathrm{{fov}}: {:.3f} \times {:.3f} ^{{\circ}}$'
                   .format(phiSpan, thetaSpan), fontsize=14)
    ax2.set_ylabel(r'$\boldsymbol{\theta}\>\>\mathrm{[deg]}$', fontsize=14)
    ax3.set_xlabel(r'$\boldsymbol{{\phi}}\>\>\mathrm{{[deg]}}$'+'\n'+\
                   r'$\mathrm{{fov}}: {:.3f} \times {:.3f} ^{{\circ}}$'
                   .format(phiSpanRot, thetaSpanRot), fontsize=14)
    ax3.set_ylabel(r'$\boldsymbol{\theta}\>\mathrm{[deg]}$', fontsize=14)
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax2.invert_xaxis()
    ax3.invert_xaxis()
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax2.set_title(r'$\mathrm{Density}$')
    ax3.set_title(r'$\mathrm{Convergence}$')
    plt.tight_layout()
    plt.show()
