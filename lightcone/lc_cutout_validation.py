import struct
import glob
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def downsample(arr, factor=0.01):
    return np.random.choice(arr, int(len(arr)*factor), replace=False)

def makePlot(outDir, downsFrac = 0.01):
    
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
    a = np.array([])
    theta = np.array([])
    phi = np.array([])
    thetaRot = np.array([])
    phiRot = np.array([])
    conv = (np.pi / 180)

    f = plt.figure(0)
    ax1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=2, projection='3d')
    ax2 = plt.subplot2grid((3,2), (2,0))
    ax3 = plt.subplot2grid((3,2), (2,1))
    
    for i in range(len(steps)):
        
        step = steps[i]
        xf = "{0}/{1}{2}/x.{2}.bin".format(outDir, prefix, step)
        yf = "{0}/{1}{2}/y.{2}.bin".format(outDir, prefix, step)
        zf = "{0}/{1}{2}/z.{2}.bin".format(outDir, prefix, step)
        af = "{0}/{1}{2}/a.{2}.bin".format(outDir, prefix, step)
        tf = "{0}/{1}{2}/theta.{2}.bin".format(outDir, prefix, step)
        pf = "{0}/{1}{2}/phi.{2}.bin".format(outDir, prefix, step)
        tRotf = "{0}/{1}{2}/thetaRot.{2}.bin".format(outDir, prefix, step)
        pRotf = "{0}/{1}{2}/phiRot.{2}.bin".format(outDir, prefix, step) 

        if(len(np.fromfile(xf, dtype=dt)) == 0):
            continue
        
        downs = downsample(np.arange(len(np.fromfile(xf, dtype=dt))), downsFrac)
        x = np.hstack([x, np.fromfile(xf, dtype=dt)[downs]])
        y = np.hstack([y, np.fromfile(yf, dtype=dt)[downs]])
        z = np.hstack([z, np.fromfile(zf, dtype=dt)[downs]])
        a = np.hstack([a, np.fromfile(af, dtype=dt)[downs]])
        theta = np.hstack([theta, np.fromfile(tf, dtype=dt)[downs]/3600])
        phi = np.hstack([phi, np.fromfile(pf, dtype=dt)[downs]/3600])
        thetaRot = np.hstack([thetaRot, np.fromfile(tRotf, dtype=dt)[downs]/3600])
        phiRot = np.hstack([phiRot, np.fromfile(pRotf, dtype=dt)[downs]/3600])

    r = np.sqrt(x**2 + y**2 + z**2)
    xRot = r*np.sin(thetaRot*conv)*np.cos(phiRot*conv)
    yRot = r*np.sin(thetaRot*conv)*np.sin(phiRot*conv)
    zRot = r*np.cos(thetaRot*conv)
    
    thetaSpan = np.max(theta) - np.min(theta) 
    phiSpan = np.max(phi) - np.min(phi)
    thetaSpanRot = np.max(thetaRot) - np.min(thetaRot) 
    phiSpanRot = np.max(phiRot) - np.min(phiRot)
    
    s1 = ax1.scatter(x, y, z, c=((1/a)-1), s=2, cmap='plasma')
    ax1.scatter(xRot, yRot, zRot, c=((1/a)-1), s=2, cmap='plasma')
    f.colorbar(s1)
    
    ax2.scatter(phi, theta, c=((1/a)-1), s=1, cmap='plasma')
    ax3.scatter(phiRot, thetaRot, c=((1/a)-1), s=1, cmap='plasma')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.set_xlabel('phi \n fov: {:.3f} x {:.3f} deg'.format(phiSpan, thetaSpan))
    ax2.set_ylabel('theta')
    ax3.set_xlabel('phi \n fov: {:.3f} x {:.3f} deg'.format(phiSpanRot, thetaSpanRot))
    ax3.set_ylabel('theta')
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax2.invert_xaxis()
    ax3.invert_xaxis()
    ax2.set_xlim([np.min(phi) - 0.2, np.max(phi) + 0.2])
    ax2.set_ylim([np.min(theta) - 0.2, np.max(theta) + 0.2])
    ax3.set_xlim([np.min(phiRot) - 0.2, np.max(phiRot) + 0.2])
    ax3.set_ylim([np.min(thetaRot) - 0.2, np.max(thetaRot) + 0.2])
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax2.set_title('original')
    ax3.set_title('to equator')
    plt.tight_layout()
    plt.show()
