from dtk import gio
from dtk import sort
import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
This file contains functions for inspecting the lightcone output
'''

def saveParticlePathData(diffRange = 'max', plot=True):
    '''
    This function loads particle from the lightcone output, and inspects the 
    difference in position resulting from the extrapolation and interpolation
    routines. This difference, per particle, is called diffVals. It then plots 
    the 3 dimensional paths of 10 of these particles - either the 10 with the 
    largest diffVals, the median, or minimum. 
    Overplotted is the path of the particle from surrounding snapshots, where
    the particle from the lightocone is matched to the snapshots by id.
    This code is meant to be run on Datastar.

    :param diffRange: whether to use the 'max', 'med'(median) or 'min' diffVals
    :param plot: whether or not to plot the particle paths. If not, then save the 
                 x,y,z data of the 10 particle paths, from the extrapolated
                 lightcone, interpolated lightcone, and snapshots
    '''

    epath = "/home/jphollowed/data/hacc/alphaQ/downsampled_particle_extrp_lc"
    ipath = "/home/jphollowed/data/hacc/alphaQ/downsampled_particle_intrp_lc"
    spath = "/home/jphollowed/data/hacc/alphaQ/downsampled_particles"

    coord = 'x'
    coord2 = 'y'
    coord3 = 'z'
    s_coord = 'x'
    s_coord2 = 'y'
    s_coord3 = 'z'

    print("Reading interpolation files")
    #iid1 = gio.gio_read("{}/lc_intrp_output_d.432".format(ipath), 'id')
    iid2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'id')
    ix2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), coord)
    iy2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), coord2)
    iz2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), coord3)
    ia2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'a')
    ivx2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'v{}'.format(coord))
    irot2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'rotation')

    print("Reading extrapolation files")
    #eid1 = gio.gio_read("{}/lc_output_d.432".format(epath), 'id')
    eid2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'id')
    ex2 = gio.gio_read("{}/lc_output_d.442".format(epath), coord)
    ey2 = gio.gio_read("{}/lc_output_d.442".format(epath), coord2)
    ez2 = gio.gio_read("{}/lc_output_d.442".format(epath), coord3)
    ea2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'a')
    evx2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'v{}'.format(coord))
    erot2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'rotation')

    print(irot2)
    print(erot2)

    print('finding unique')
    iun = np.unique(iid2, return_counts=True)
    eun = np.unique(eid2, return_counts=True)
    if(max(iun[1]) > 1 or max(eun[1]) > 1): 
        #pdb.set_trace()
        pass
    iunMask = np.ones(len(iid2), dtype=bool)
    iunMask[np.where(np.in1d(iid2, iun[0][iun[1] > 1]))[0]] = 0

    print('get intersecting particles')
    intersection_itoe = np.in1d(iid2[iunMask], eid2)
    intersection_etoi = np.in1d(eid2, iid2[iunMask])

    print('sorting array 2')
    eSort = np.argsort(eid2[intersection_etoi])

    print('matching arrays')
    matchMap = sort.search_sorted(eid2[intersection_etoi], 
                                  iid2[iunMask][intersection_itoe], sorter=eSort)

    iMask = np.linspace(0, len(iid2)-1, len(iid2), dtype=int)[iunMask][intersection_itoe]
    eMask = np.linspace(0, len(eid2)-1, len(eid2), dtype=int)[intersection_etoi][matchMap]

    print('diffing')
    xdiff = np.abs(ix2[iMask] - ex2[eMask])
    ydiff = np.abs(iy2[iMask] - ey2[eMask])
    zdiff = np.abs(iz2[iMask] - ez2[eMask])
    magDiff = np.linalg.norm(np.array([xdiff, ydiff, zdiff]).T, axis=1)
    
    f = plt.figure(0)
    ax =  f.add_subplot(111)
    ax.hist(magDiff, 1000)
    ax.set_yscale('log')
    plt.show()

    if(diffRange == 'max'):
        diffVals = np.argsort(magDiff)[::-1][0:10]
        savePath = "lc_particle_paths/max_diff"
    if(diffRange == 'med'):
        diffVals = np.argsort(magDiff)[::-1][len(xdiff)/2:len(xdiff)/2 + 20][0:10]
        savePath = "lc_particle_paths/med_diff"
    if(diffRange == 'min'):
        diffVals = np.argsort(magDiff)[0:10]
        savePath = "lc_particle_paths/min_diff"

    print("Reading timestep files")
    sid0 = gio.gio_read("{}/m000.mpicosmo.421".format(spath), 'id')
    sx0 = gio.gio_read("{}/m000.mpicosmo.421".format(spath), s_coord)
    sy0 = gio.gio_read("{}/m000.mpicosmo.421".format(spath), s_coord2)
    sz0 = gio.gio_read("{}/m000.mpicosmo.421".format(spath), s_coord3)

    sid1 = gio.gio_read("{}/m000.mpicosmo.432".format(spath), 'id')
    sx1 = gio.gio_read("{}/m000.mpicosmo.432".format(spath), s_coord)
    sy1 = gio.gio_read("{}/m000.mpicosmo.432".format(spath), s_coord2)
    sz1 = gio.gio_read("{}/m000.mpicosmo.432".format(spath), s_coord3)

    sid2 = gio.gio_read("{}/m000.mpicosmo.442".format(spath), 'id')
    sx2 = gio.gio_read("{}/m000.mpicosmo.442".format(spath), s_coord)
    sy2 = gio.gio_read("{}/m000.mpicosmo.442".format(spath), s_coord2)
    sz2 = gio.gio_read("{}/m000.mpicosmo.442".format(spath), s_coord3)

    sid3 = gio.gio_read("{}/m000.mpicosmo.453".format(spath), 'id')
    sx3 = gio.gio_read("{}/m000.mpicosmo.453".format(spath), s_coord)
    sy3 = gio.gio_read("{}/m000.mpicosmo.453".format(spath), s_coord2)
    sz3 = gio.gio_read("{}/m000.mpicosmo.453".format(spath), s_coord3)

    sid4 = gio.gio_read("{}/m000.mpicosmo.464".format(spath), 'id')
    sx4 = gio.gio_read("{}/m000.mpicosmo.464".format(spath), s_coord)
    sy4 = gio.gio_read("{}/m000.mpicosmo.464".format(spath), s_coord2)
    sz4 = gio.gio_read("{}/m000.mpicosmo.464".format(spath), s_coord3)

    for i in range(len(diffVals)):

        idx = diffVals[i]
        print('Matching to snapshots for idx {} with diff of {}'.format(idx, magDiff[idx]))
        print('Particle ID is {}'.format(iid2[iMask][idx]))
        ix = ix2[iMask][idx]
        iy = iy2[iMask][idx]
        iz = iz2[iMask][idx]
        ia = ia2[iMask][idx]
        
        ex = ex2[eMask][idx]
        ey = ey2[eMask][idx]
        ez = ez2[eMask][idx]
        ea = ea2[eMask][idx]

        s0_idx = np.where(sid0 == iid2[iMask][idx])
        s1_idx = np.where(sid1 == iid2[iMask][idx])
        s2_idx = np.where(sid2 == iid2[iMask][idx])
        s3_idx = np.where(sid3 == iid2[iMask][idx])
        s4_idx = np.where(sid4 == iid2[iMask][idx])
      
        sxi0 = sx0[s0_idx][0]
        sxi1 = sx1[s1_idx][0]
        sxi2 = sx2[s2_idx][0]
        sxi3 = sx3[s3_idx][0]
        sxi4 = sx4[s4_idx][0]
        
        syi0 = sy0[s0_idx][0]
        syi1 = sy1[s1_idx][0]
        syi2 = sy2[s2_idx][0]
        syi3 = sy3[s3_idx][0]
        syi4 = sy4[s4_idx][0]
        
        szi0 = sz0[s0_idx][0]
        szi1 = sz1[s1_idx][0]
        szi2 = sz2[s2_idx][0]
        szi3 = sz3[s3_idx][0]
        szi4 = sz4[s4_idx][0]
        
        aa = np.linspace(1/(1+200), 1, 500)
        sai0 = aa[422]
        sai1 = aa[433]
        sai2 = aa[443]
        sai3 = aa[454]
        sai4 = aa[465]
        
        truex = np.array([sxi0, sxi1, sxi2, sxi3, sxi4])
        truey = np.array([syi0, syi1, syi2, syi3, syi4])
        truez = np.array([szi0, szi1, szi2, szi3, szi4])
        truea = np.array([sai0, sai1, sai2, sai3, sai4])
        interpolx = np.array([sxi2, ix])
        interpoly = np.array([syi2, iy])
        interpolz = np.array([szi2, iz])
        interpola = np.array([sai2, ia])
        extrapx = np.array([sxi2, ex])
        extrapy = np.array([syi2, ey])
        extrapz = np.array([szi2, ez])
        extrapa = np.array([sai2, ea])
       
        if(plot == 0):
            np.save('{}/ix_{}.npy'.format(savePath, i), interpolx)
            np.save('{}/iy_{}.npy'.format(savePath, i), interpoly)
            np.save('{}/iz_{}.npy'.format(savePath, i), interpolz)
            np.save('{}/ia_{}.npy'.format(savePath, i), interpola)
            np.save('{}/iid_{}.npy'.format(savePath, i), iid2[iMask][idx])
            
            np.save('{}/ex_{}.npy'.format(savePath, i), extrapx)
            np.save('{}/ey_{}.npy'.format(savePath, i), extrapy)
            np.save('{}/ez_{}.npy'.format(savePath, i), extrapz)
            np.save('{}/ea_{}.npy'.format(savePath, i), extrapa)

            np.save('{}/truex_{}.npy'.format(savePath, i), truex)
            np.save('{}/truey_{}.npy'.format(savePath, i), truey)
            np.save('{}/truez_{}.npy'.format(savePath, i), truez)
            np.save('{}/truea_{}.npy'.format(savePath, i), truea)
            print("saved {} for particle {}".format(diffRange, i))
        else:
            f = plt.figure(0)
            ax = f.add_subplot(211)
            ax.plot(truex, truea, '.-g')
            ax.plot(interpolx, interpola, '.-m')
            ax.plot(extrapx, extrapa, '.-r')
            ax2 = f.add_subplot(212, projection='3d')
            ax2.plot(truex, truey, truez, '.-g')
            ax2.plot(interpolx, interpoly, interpolz, '.-m')
            ax2.plot(extrapx, extrapy, extrapz, '.-r')
            ax2.plot([truex[0]], [truey[0]], [truez[0]], '*b')
            ax2.set_xlabel(coord)
            ax2.set_ylabel(coord2)
            ax2.set_zlabel(coord3)
            plt.show()
