import numpy as np
import pdb
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import matplotlib.ticker as plticker
import h5py as h5
import glob

'''
This file contains functions for inspecting the lightcone output
'''
def config(cmap): 
    
    rcParams.update({'figure.autolayout': True})
    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    rcParams.update(params)
    colors = cmap(np.linspace(0.2, 0.8, 3))
    c = cycler('color', colors)
    plt.rcParams["axes.prop_cycle"] = c
    

def saveParticlePathData(diffRange='max', plot=True, posDiffOnly=False, dt_type='linear'):
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
    from dtk import gio
    from dtk import sort
    
    config(cmap=plt.cm.plasma)

    epath="/home/jphollowed/data/hacc/alphaQ/lightcone/downsampled_particle_extrp_lc/step442_{}".format(dt_type)
    ipath="/home/jphollowed/data/hacc/alphaQ/lightcone/downsampled_particle_intrp_lc/step442_{}".format(dt_type)
    spath="/home/jphollowed/data/hacc/alphaQ/particles/downsampled_particles"
   
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
    ivy2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'v{}'.format(coord2))
    ivz2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'v{}'.format(coord3))
    irot2 = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'rotation')

    print("Reading extrapolation files")
    #eid1 = gio.gio_read("{}/lc_output_d.432".format(epath), 'id')
    eid2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'id')
    ex2 = gio.gio_read("{}/lc_output_d.442".format(epath), coord)
    ey2 = gio.gio_read("{}/lc_output_d.442".format(epath), coord2)
    ez2 = gio.gio_read("{}/lc_output_d.442".format(epath), coord3)
    ea2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'a')
    evx2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'v{}'.format(coord))
    evy2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'v{}'.format(coord2))
    evz2 = gio.gio_read("{}/lc_output_d.442".format(epath), 'v{}'.format(coord3))
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
    xdiff = ix2[iMask] - ex2[eMask]
    ydiff = iy2[iMask] - ey2[eMask]
    zdiff = iz2[iMask] - ez2[eMask]
    posDiff = np.linalg.norm(np.array([xdiff, ydiff, zdiff]).T, axis=1)
    
    redshiftDiff = np.abs(((1/ia2)-1)[iMask] - ((1/ea2)-1)[eMask])

    vxdiff = ivx2[iMask] - evx2[eMask]
    vydiff = ivy2[iMask] - evy2[eMask]
    vzdiff = ivz2[iMask] - evz2[eMask]
    mag_vDiff = np.linalg.norm(np.array([vxdiff, vydiff, vzdiff]).T, axis=1)
   
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bins = 300
    f = plt.figure(0)
    
    ax =  f.add_subplot(311)
    ax.hist(posDiff, bins, color=colors[0])
    ax.set_yscale('log')
    ax.set_xlabel(r'$\left|\vec{r}_\mathrm{extrap} - \vec{r}_\mathrm{interp}\right|\>\>\mathrm{(Mpc/h)}$', fontsize=18)
    
    ax2 =  f.add_subplot(312)
    ax2.hist(mag_vDiff, bins, color=colors[1])
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\left|\vec{v}_\mathrm{extrap} - \vec{v}_\mathrm{interp}\right| \>\>\mathrm{(km/s)}$', fontsize=18)
    
    ax3 =  f.add_subplot(313)
    ax3.hist(redshiftDiff, bins, color=colors[2])
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$\left|z_\mathrm{extrap} - z_\mathrm{interp}\right|$', fontsize=18)
    
    plt.show()
    if(posDiffOnly): return

    print('matching to specified range ({})'.format(diffRange))
    duplicateRanges = ['weird', 'duplShared', 'dupl_intrpUnique', 'dupl_extrpUnique']
    if(diffRange == 'max'):
        diffVals = np.argsort(posDiff)[::-1][0:10]
        savePath = "lc_particle_paths/max_diff"
    if(diffRange == 'med'):
        diffVals = np.argsort(posDiff)[::-1][len(xdiff)/2:len(xdiff)/2 + 20][0:10]
        savePath = "lc_particle_paths/med_diff"
    if(diffRange == 'min'):
        diffVals = np.argsort(posDiff)[0:10]
        savePath = "lc_particle_paths/min_diff"
    if(diffRange == 'dupl'):
        dups_intrp = h5.File(
                 '/home/jphollowed/code/lc_duplicates/output/dups_interp.hdf5','r')
        dupIds = dups_intrp['id'][:] 
        iMask = iMask[dupMask]
        eMask = eMask[dupMask]
        posDiff = posDiff[dupMask]
        diffVals = np.argsort(posDiff)[::-1][0:20]
        savePath = "lc_particle_paths/dupl_diff" 
    if(diffRange in duplicateRanges):
        badIds = np.load('{}Ids.npy'.format(diffRange))
        iMask = np.ones(len(iid2), dtype=bool)
        diffVals = np.where(np.in1d(iid2[iMask], badIds))[0][0:20]
        savePath = "lc_particle_paths/{}_ids".format(diffRange)


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
        if(diffRange != 'weird' and 'Unique' not in diffRange):
            print('Matching to snapshots for idx {} with diff of {}'.format(idx,posDiff[idx]))
        else:
            print('Matching to snapshots for idx {} with diff of NA'.format(idx))
        print('Particle ID is {}'.format(iid2[iMask][idx]))
        
        
        if(diffRange != 'weird' and diffRange != 'dupl_extrapUnique'):
            ix = ix2[iMask][idx]
            iy = iy2[iMask][idx]
            iz = iz2[iMask][idx]
            ia = ia2[iMask][idx]
       
        if(diffRange != 'weird' and diffRange != 'dupl_intrpUnique'):
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
        
        if(diffRange != 'weird' and diffRange != 'dupl_extrpUnique'):
            interpolx = np.array([sxi2, ix])
            interpoly = np.array([syi2, iy])
            interpolz = np.array([szi2, iz])
            interpola = np.array([sai2, ia])

        if(diffRange != 'weird' and diffRange != 'dupl_intrpUnique'):
            extrapx = np.array([sxi2, ex])
            extrapy = np.array([syi2, ey])
            extrapz = np.array([szi2, ez])
            extrapa = np.array([sai2, ea])
       
        if(plot == 0):
            
            if(diffRange != 'weird' and diffRange != 'dupl_extrpUnique'):
                np.save('{}/ix_{}.npy'.format(savePath, i), interpolx)
                np.save('{}/iy_{}.npy'.format(savePath, i), interpoly)
                np.save('{}/iz_{}.npy'.format(savePath, i), interpolz)
                np.save('{}/ia_{}.npy'.format(savePath, i), interpola)
                np.save('{}/iid_{}.npy'.format(savePath, i), iid2[iMask][idx])
            
            if(diffRange != 'weird' and diffRange != 'dupl_intrpUnique'):
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
            if(diffRange != 'weird'): ax2.plot(extrapx, extrapy, extrapz, '.-r')
            ax2.plot([truex[0]], [truey[0]], [truez[0]], '*b')
            ax2.set_xlabel(coord)
            ax2.set_ylabel(coord2)
            ax2.set_zlabel(coord3)
            plt.show()


def plotParticlePaths(diffRange = 'max'):
    '''
    This function plots the same thing as the one above. Instead of doing the particle
    matching and position diffing, however, it reads the outpt of saveParticlePathData()
    in the case that that function was run with plot=False. This is to enable local 
    plotting for better 3d display without having to be on datastar
    '''
    config(cmap=plt.cm.cool)

    path = '/home/joe/gdrive2/work/HEP/data/hacc/alphaQ/lightcone/lc_particle_paths'
    duplicateRanges = ['weird', 'duplShared', 'dupl_intrpUnique', 'dupl_extrpUnique']
    data = '{}/{}_diff'.format(path, diffRange)
    if(diffRange in duplicateRanges): 
        data = '{}/{}_ids'.format(path, diffRange)
    numFiles = len(glob.glob('{}/truex_*'.format(data)))

    for i in range(numFiles):

        if(diffRange != 'dupl_extrpUnique'):
            ix = np.load('{}/ix_{}.npy'.format(data, i))
            iy = np.load('{}/iy_{}.npy'.format(data, i))
            iz = np.load('{}/iz_{}.npy'.format(data, i))
            ia = np.load('{}/ia_{}.npy'.format(data, i))
            iid = np.load('{}/iid_{}.npy'.format(data, i))
        
        if(diffRange != 'weird' and diffRange != 'dupl_intrpUnique'):
            ex = np.load('{}/ex_{}.npy'.format(data, i))
            ey = np.load('{}/ey_{}.npy'.format(data, i))
            ez = np.load('{}/ez_{}.npy'.format(data, i))
            ea = np.load('{}/ea_{}.npy'.format(data, i))
        
        truex = np.load('{}/truex_{}.npy'.format(data, i))
        truey = np.load('{}/truey_{}.npy'.format(data, i))
        truez = np.load('{}/truez_{}.npy'.format(data, i))
        truea = np.load('{}/truea_{}.npy'.format(data, i))

        ax = plt.subplot2grid((3,3), (0,0), rowspan=2, colspan=2, projection='3d')
        x = np.random.randn(10)
        y = np.random.randn(10)
        z = np.random.randn(10)
        
        ax.plot(truex, truey, truez, '--k.')
        
        if(diffRange != 'weird' and diffRange != 'dupl_intrpUnique'):
            ax.plot(ex, ey, ez, '-o', lw=2)
        
        ax.plot([truex[0]], [truey[0]], [truez[0]], '*', ms=10)
        
        if(diffRange != 'dupl_extrpUnique'):
            ax.plot(ix, iy, iz, '-o', lw=2)
            plt.title(r'$\mathrm{{ ID }}\>\>{}$'.format(iid), y=1.08, fontsize=18)
        
        ax.set_xlabel(r'$x\>\>\mathrm{(Mpc/h)}$', fontsize=12, labelpad=12)
        ax.set_ylabel(r'$y\>\>\mathrm{(Mpc/h)}$', fontsize=12, labelpad=12)
        ax.set_zlabel(r'$z\>\>\mathrm{(Mpc/h)}$', fontsize=12, labelpad=12)
        ax.tick_params(axis='both', which='major', labelsize=8)
        for t in range(len(ax.xaxis.get_major_ticks())): 
            if(t%2 == 1): 
                ax.xaxis.get_major_ticks()[t].label.set_color([1, 1, 1]) 
                ax.xaxis.get_major_ticks()[t].label.set_fontsize(0) 
        for t in range(len(ax.yaxis.get_major_ticks())): 
            if(t%2 == 1): 
                ax.yaxis.get_major_ticks()[t].label.set_color([1, 1, 1]) 
                ax.yaxis.get_major_ticks()[t].label.set_fontsize(0) 
        for t in range(len(ax.zaxis.get_major_ticks())): 
            if(t%2 == 1): 
                ax.zaxis.get_major_ticks()[t].label.set_color([1, 1, 1]) 
                ax.zaxis.get_major_ticks()[t].label.set_fontsize(0) 
        
        ax_xa = plt.subplot2grid((3,3), (2,0), colspan=2)
        ax_xa.plot(truex, (1/truea)-1, '--k.')
        
        if(diffRange != 'weird' and diffRange != 'dupl_intrpUnique'):
            ax_xa.plot(ex, (1/ea)-1, '-o', lw=2)
        
        ax_xa.plot(truex[0], (1/truea[0])-1, '*', ms=10)
        
        if(diffRange != 'dupl_extrpUnique'):
            ax_xa.plot(ix, (1/ia)-1, '-o', lw=2)
        
        ax_xa.set_xlabel(r'$x\>\>\mathrm{(Mpc/h)}$', fontsize=14, labelpad=6)
        ax_xa.set_ylabel(r'$\mathrm{redshift}$', fontsize=14, labelpad=6)
        ax_xa.set_yticks((1/truea)-1)
        ax_xa.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_xa.yaxis.tick_right()
        ax_xa.yaxis.set_label_position("right")
        ax_xa.invert_yaxis()
        ax_xa.grid()


        ax_za = plt.subplot2grid((3,3), (0,2), rowspan=2)
        ax_za.plot((1/truea)-1, truez, '--k.', label='true path')
        
        if(diffRange != 'weird' and diffRange != 'dupl_intrpUnique'):
            ax_za.plot((1/ea)-1, ez, '-o', lw=2, label = 'extrapolation')
        
        ax_za.plot((1/truea[0])-1, truez[0], '*', ms=10, label='starting position')
        
        if(diffRange != 'dupl_extrpUnique'):
            ax_za.plot((1/ia)-1, iz, '-o', lw=2, label='interpolation')
        
        ax_za.set_ylabel(r'$z\>\>\mathrm{(Mpc/h)}$', fontsize=14, labelpad=6)
        ax_za.set_xlabel(r'$\mathrm{redshift}$', fontsize=14, labelpad=6)
        ax_za.set_xticks(1/(truea)-1)
        for tick in ax_za.get_xticklabels(): tick.set_rotation(90)
        ax_za.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_za.yaxis.tick_right()
        ax_za.yaxis.set_label_position("right")
        ax_za.grid()
        ax_za.legend(bbox_to_anchor=(1.12, -0.35))


        plt.gcf().set_size_inches(8, 6)
        plt.gcf().tight_layout()

        plt.gcf().canvas.manager.window.move(540, 200)
        plt.show()




def compareDuplicates(dt_type = 'linear'):

    path = '/home/joe/gdrive2/work/HEP/data/hacc/alphaQ/lightcone/lc_duplicates'
    idupl = h5.File('{}/dups_interp_{}.hdf5'.format(path, dt_type), 'r')
    edupl = h5.File('{}/dups_extrap_{}.hdf5'.format(path, dt_type), 'r')
    dslc = np.load('lc_intrp_output_tinySample.npz')

    print('Duplicate fraction for old output: {}'.format(edupl['repeat_frac'][:][0]))
    print('Duplicate fraction for new output: {}'.format(idupl['repeat_frac'][:][0]))
    
    f = plt.figure(0)
    axe = f.add_subplot(121)
    axi = f.add_subplot(122)

    maski = np.in1d(idupl['id'], edupl['id'])
    maske = np.in1d(edupl['id'], idupl['id'])

    #axi.plot(dslc['x'], dslc['y'], '.y', ms=1)
    #axe.plot(dslc['x'], dslc['y'], '.y', ms=1)
    
    pdb.set_trace()
    axe.plot(edupl['x'], edupl['y'], '.g', ms=1)
    axe.plot(edupl['x'][~maske], edupl['y'][~maske], '+m', mew=1)
    axe.set_xlabel('x (Mpc/h)')
    axe.set_ylabel('y (Mpc/h)')

    axi.plot(idupl['x'], idupl['y'], '.b', ms=1)
    axi.plot(idupl['x'][~maski], idupl['y'][~maski], '+r', mew=1)
    axi.set_xlabel('x (Mpc/h)')
    axi.set_ylabel('y (Mpc/h)')

    distMask = np.linalg.norm(np.array([idupl['x'][:], idupl['y'][:]]).T, axis=1) < 255
    print(np.sum(distMask))

    axi.plot(idupl['x'][distMask], idupl['y'][distMask], 'xk', mew=1)
    
    np.save('weirdIds.npy', idupl['id'][distMask])
    
    dupl_intrpUnique = np.logical_and(~distMask, ~maski)
    
    np.save('duplSharedIds.npy', idupl['id'][maski])
    np.save('dupl_intrpUniqueIds.npy', idupl['id'][dupl_intrpUnique])
    np.save('dupl_extrpUniqueIds.npy', edupl['id'][~maske])
 
    plt.show()


def downsampleOutput():
    
    from dtk import gio
    from dtk import sort 

    ipath="/home/jphollowed/data/hacc/alphaQ/lightcone/downsampled_particle_intrp_lc/step442"
   
    coord = 'x'
    coord2 = 'y'
    coord3 = 'z'
    s_coord = 'x'
    s_coord2 = 'y'
    s_coord3 = 'z'

    print("Reading interpolation files")
    iid = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'id')
    ix = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), coord)
    iy = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), coord2)
    iz = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), coord3)
    irot = gio.gio_read("{}/lc_intrp_output_d.442".format(ipath), 'rotation')
    
    idx = np.linspace(0, len(iid)-1, len(iid), dtype=int)
    randIdx = np.random.choice(idx, size=100000, replace=False)

    ds_iid = iid[randIdx]
    ds_ix = ix[randIdx]
    ds_iy = iy[randIdx]
    ds_iz = iz[randIdx]
    ds_irot = irot[randIdx]

    np.savez('lc_intrp_output_tinySample.npz', id=ds_iid, x=ds_ix, y=ds_iy, 
                                               z=ds_iz,rot=ds_irot)
    return;

def findDuplicates(lc_type = 'i', dt_type='nonlin'):

    from dtk import sort
    from dtk import gio

    print('reading data')
    if(lc_type == 'i'):
        path1="/home/jphollowed/data/hacc/alphaQ/lightcone/"\
              "downsampled_particle_intrp_lc/step442_{}".format(dt_type)
        path2="/home/jphollowed/data/hacc/alphaQ/lightcone/"\
              "downsampled_particle_intrp_lc/step432_{}".format(dt_type)
        file1 = "{}/lc_intrp_output_d.442".format(path1)
        file2 = "{}/lc_intrp_output_d.432".format(path2)
        outfile = h5.File('dups_interp_{}.hdf5'.format(dt_type), 'w')
    if(lc_type == 'e'):
        path1="/home/jphollowed/data/hacc/alphaQ/lightcone/"\
              "downsampled_particle_extrp_lc/step442_{}".format(dt_type)
        path2="/home/jphollowed/data/hacc/alphaQ/lightcone/"\
              "downsampled_particle_extrp_lc/step432_{}".format(dt_type)
        file1 = "{}/lc_output_d.442".format(path1)
        file2 = "{}/lc_output_d.432".format(path2)
        outfile = h5.File('dups_extrap_{}.hdf5'.format(dt_type), 'w')


    ids1 = gio.gio_read(file1, 'id')
    ids2 = gio.gio_read(file2, 'id')
    
    print('matching')
    matches = sort.search_sorted(ids1, ids2)

    matchesMask2 = matches != -1
    matchesMask1 = matches[matchesMask2]

    print('found {} duplicates'.format(np.sum(matchesMask2)))

    dup_ids1 = ids1[matchesMask1]
    x1 = gio.gio_read(file1, 'x')[matchesMask1]
    y1 = gio.gio_read(file1, 'y')[matchesMask1]
    z1 = gio.gio_read(file1, 'z')[matchesMask1]
    
    dup_ids2 = ids2[matchesMask2]
    x2 = gio.gio_read(file2, 'x')[matchesMask2]
    y2 = gio.gio_read(file2, 'y')[matchesMask2]
    z2 = gio.gio_read(file2, 'z')[matchesMask2]

    repeat_frac = float(len(dup_ids2)) / len(ids2) 
    print('repeat fraction is {}'.format(repeat_frac))

    print('writing out')
    outfile.create_dataset('repeat_frac', data=np.array([repeat_frac]))
    outfile.create_dataset('id', data = np.hstack([dup_ids2, dup_ids1]))
    outfile.create_dataset('x', data = np.hstack([x2, x1]))
    outfile.create_dataset('y', data = np.hstack([y2, y1]))
    outfile.create_dataset('z', data = np.hstack([z2, z1]))
    

    
