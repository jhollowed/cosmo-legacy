import struct
import glob
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import scipy.stats as st
from KDEpy import FFTKDE

def downsample(arr, factor=0.01, replace=False):
    '''
    Downsamaple an array

    Params:
    arr: numpy array
    factor: retain this percentage of the values
    '''
    return np.random.choice(arr, int(len(arr)*factor), replace=replace)


def visCutout(outDir, bins=50, showBeams=False, downsFrac = 0.10, cm='plasma'):
    '''
    Visualize a lightcone cutout. At least two plots will be generated, each showing
    the cutout in projected angular coordinates with respect to the observer (origin).
    Optionally, a third 3d plot can be produced which shows the full cutout in comoving
    cartesian coordinates.

    Params:
    :outDir: The top-level cutout output directory to read from
    :bins: how many bins per-dimension to plot the projected density
    :showBeams: if True, 3d plot of the cutout in comoving cartesian space is produced
    :downsFrac: the particle downsampling fraction
    :cm: the colormap to use (coresponding to density in the projected plots, and to 
         redshift in the 3d plot if showBeams==True)
    '''

    mpl.rcParams.update({'figure.autolayout': True})
    params = {'text.usetex': True, 'mathtext.fontset': 'stixsans'}
    mpl.rcParams.update(params)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    
    subdirs = np.array(glob.glob("{}/*".format(outDir)))
    subdirs = subdirs[np.array([s.split('.')[-1]!='csv' for s in subdirs])].tolist()
    
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue

    steps = [int(s.split('Cutout')[-1]) for s in subdirs]
    steps = sorted(steps)
    print('found steps {}'.format(steps))
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
        if(step < 300): continue
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
        
        # for displaying each shell individually
        #plt.Figure()
        #bins=70
        #plt.hist2d(np.fromfile(pf, dtype=dt), np.fromfile(tf, dtype=dt), bins, norm=mpl.colors.LogNorm(), cmap=cm)
        #plt.title(step)
        #plt.show()

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
    ax2.set_title(r'$\mathrm{Density (rotation applied)}$')
    ax3.set_title(r'$\mathrm{Density (original position)}$')
    plt.tight_layout()
    plt.show()


def fov_kde(outDir, gp=100, N=10, steps=None, downsFrac = 0.10, cm='plasma'):
    '''
    Visualize a lightcone cutout by applying a Gaussian kernel density estiamtor to a particle
    distribution in theta-phi space.

    Params:
    :outDir: The top-level cutout output directory to read from
    :gp: how many grid points on which to evaluate the gaussian kde
    :step: which shell to include in the plotting. If an integer, use that shell. 
           If a list of integers, use those shells. If None, use all shells. If 
           the string 'halo' is passed, assume a properties.csv file is present in
           outDir, and read the shell number containing the target halo as the value
           labeled 'lc_shell'
    :downsFrac: the particle downsampling fraction
    :cm: the colormap to use
    '''
    
    # get subdirs names/prefix
    subdirs = np.array(glob.glob("{}/*".format(outDir)))
    subdirs = subdirs[np.array([s.split('.')[-1]!='csv' for s in subdirs])].tolist()
    
    for i in range(len(subdirs[0].split('/')[-1])):
        try:
            (int(subdirs[0].split('/')[-1][i]))
            prefix = subdirs[0].split('/')[-1][0:i]
            break
        except ValueError:
            continue

    # get shells of interest
    if(steps is not None):
        if steps == 'halo':
            steps = [int(np.genfromtxt('{}/properties.csv'.format(outDir), delimiter=',')[1])]
    else:
        steps = [s.split('Cutout')[-1] for s in glob.glob("{}/*Cutout*".format(outDir))]
    print('shells to include = {}'.format(steps))

    # read theta, phi
    print('reading {}'.format(outDir.split('/')[-1]))
    t = np.array([])
    p = np.array([])
    for step in steps:
        t = np.hstack([t, np.fromfile('{0}/{2}{1}/theta.{1}.bin'.format(outDir, step, prefix), '<f')])
        p = np.hstack([p, np.fromfile('{0}/{2}{1}/phi.{1}.bin'.format(outDir, step, prefix), '<f')])
    mm = downsample(np.arange(len(t)), downsFrac, replace=True)
    t = t[mm]
    p = p[mm]
    val = np.vstack([t, p]).T

    # eval kde on grid
    print('evaluating kde')
    grid, f = FFTKDE(kernel='gaussian').fit(val)((gp, gp))
    tt, pp = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    f = f - np.mean(f)
    f = f.reshape(gp, gp).T

    # vis
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    cfset = ax.contourf(tt, pp, f, cmap=cm, vmin=0)
    plt.show()
 

def plot_timing():
    '''
    Visualize the timing of a collection of cutout runs.
    ... finish these docs
    '''
    
    nodes = [32, 64, 128, 256, 512]
    nodes = [32, 64]
    halos = [16, 64, 256]
    halos=[16]

    steps = [487, 475, 464, 453, 442, 432, 421, 411, 401, 392, 382, 
             373, 365, 355, 347, 338, 331, 323, 315, 307, 300, 293, 
             286, 279, 272, 266, 259, 253, 247]

    read_times = [0] * len(nodes)
    redist_times = [0] * len(nodes)
    comp_times = [0] * len(nodes)
    write_times = [0] * len(nodes)

    for k in range(len(halos)):
        
        for j in range(len(nodes)):

            outputFile = '/home/hollowed/cutout_run_dirs/alphaQ/cutout_alphaQ_downs/N{}_R4_lc_halos_{}.txt_cutout_downs_depth_0.0-1.0.output'.format(nodes[j], halos[k])
            output = open(outputFile)
            lines = np.array(output.readlines())
            readMask = np.array(["Read time:" in line for line in lines])
            redistMask = np.array(["Redistribution time:" in line for line in lines])
            compMask = np.array(["cutout computation time:" in line for line in lines])
            writeMask = np.array(["write time:" in line for line in lines])

            #pdb.set_trace()

            read_times[j] = np.cumsum([float(s.split(': ')[-1].split(' s')[0]) for s in lines[readMask]])
            redist_times[j] = np.cumsum([float(s.split(': ')[-1].split(' s')[0]) for s in lines[redistMask]])
            comp_times[j] = np.cumsum([float(s.split(': ')[-1].split(' s')[0]) for s in lines[compMask]])
            write_times[j] = np.cumsum([float(s.split(': ')[-1].split(' s')[0]) for s in lines[writeMask]])

        f = plt.figure(k)
        ax = f.add_subplot(111)
        cmap = plt.cm.get_cmap('plasma')
        plt.title('{} halos'.format(halos[k]))

        ls = ['-', '--', '-.', ':']
        lw = [2, 2, 1, 2, 2, 0.5]
        col = list(cmap(np.linspace(0.2, 0.8, 6)))
        #steps = np.arange(len(read_times[0]))

        for j in range(len(nodes)):
          
            print(j)
            ax.plot(steps[:len(read_times[j])], read_times[j], label = 'Read in time ({} nodes)'.format(nodes[j]), linestyle=ls[0], lw=lw[j], color=col[j])
            ax.plot(steps[:len(redist_times[j])], redist_times[j], label = 'Redist. time ({} nodes)'.format(nodes[j]), linestyle=ls[1], lw=lw[j], color=col[j]) 
            
            ax.plot(
                    np.linspace(max(steps[:len(comp_times[j])]), 
                                min(steps[:len(comp_times[j])]), 
                                len(steps[:len(comp_times[j])])*halos[k]), 
                    comp_times[j], 
                    label = 'comp time ({} nodes)'.format(nodes[j]), linestyle=ls[2], lw=lw[j], color=col[j])
            
            ax.plot(
                    np.linspace(max(steps[:len(comp_times[j])]), 
                                min(steps[:len(comp_times[j])]), 
                                len(steps[:len(comp_times[j])])*halos[k]), 
                    write_times[j], 
                    label = 'Write time ({} nodes)'.format(nodes[j]), linestyle=ls[3], lw=lw[j], color=col[j])
            ax.invert_xaxis()

        plt.grid()
        plt.legend()
    plt.show()     
