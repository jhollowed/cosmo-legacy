# Joe Hollowed
# CPAC 2018

import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import pdb
import glob
import numpy as np
from mpi4py import MPI
import genericio as gio
#import lc_interpolation_validation as iv

def list_halos(lcDir, outDir, maxStep, minStep, rL, corrLength, phiMax, haloCat=None, 
               massDef = 'fof', minMass=1e14, maxMass=1e18, outFrac=0.01, numFiles=1):

    '''
    This function generates a list of halo tags, properties, and positions, in a text 
    file, in the format expected by Use Case 2 of the lightcone cutout code
    at https://github.com/jhollowed/cosmo-cutout. That is, it reads the output of a
    halo lightcone run, finds all halos above some mass cut, and writes them to
    a text file with halos printed per-row, as such:
    
    output.txt:
    tag1 redshift1 mass1 radius1 x1 y1 z1
    tag2 redshift2 mass2 radius2 x2 y2 z2
    tag3 redshift3 mass3 radius3 x3 y3 z3
    .
    .
    .
    
    where the radius column will be omitted if the input arg massDef == "fof". The tags 
    are of the form {fof_halo_tag}_{lc_replication_identifier}

    This function is written with MPI support; if run with mpirun or mpiexec, then the 
    lightcone shells found within the redshift range of interest will be distributed as 
    evenly as possible across ranks.
    
    Params:
    :param lcDir: top-level directory of a halo lightcone, where the subdirectory 
                  structure is expected to match that described in section 4.5 (fig 7)
                  of the Creating Lightcones in HACC document (step-wise subdirectories 
                  expected). It is expected that this lightcone was built using the
                  interpolation lightcone driver, and thus the 'id' field is expected
                  to contain merger tree fof tags (including fragment bits and sign).
    :para outDir: the output directory for the resultant text file
    :param maxStep: The largest (lowest redshift) lightcone shell to read in
    :param minStep: The smallest (highest redshift) lightcone shell to read in
    :param rL: The length of the simulation box underlying the input lightcone in Mpc/h
    :param corrLength: The length scale within which to consider the density field correlated 
                       with a halo, in Mpc/h-- halos within this length of a replicated box
                       boundary will experience correlation breaks in the density field
                       and will be rejected
    :param phiMax: the maximum angular scale that one may want to use on subsequent
                   halo lightcone cutout runs given the halos output by this function, 
                   in arcseconds. Halos within that angular scale from the octant boundary
                   will be rejected
    :param phiMax: The side length of the simulation underlying the input lightcone in Mpc/h
    :param haloCat: If this argument is None, then it is assumed that the input lightcone at 
                    lcDir has a valid 'mass' column. If it doesn't, then this argument should
                    point to a top-level directory of a halo catalog from which to match
                    id's and gather FOF/SO masses. Step-wise subdirectoires are expected with 
                    the form 'STEPXXX'
    :param massDef: should either be "fof", "sod", or None. If haloCat != None, then this arg 
                    specifies whether to read FOF or SO masses from the matching halo catalog, 
                    haloCat. If massDef = "sod", then also gather the r200 radii from haloCat.
                    If haloCat == None, then all this arg does is label the lightcone-provided
                    mass column as either an "fof" or "sod" mass in the output cutout meta data
    :param minMass: The minimum halo mass to write out to the text files (defaults to 1e14)
    :param maxMass: The maximum halo mass to write out to the text files (defaults to 1e18, or 
                    no upper limit)
    :param outFrac: The fraction of the identified halos to actually output
    :param numFiles: How many text files to write. That is, if 30 halos are found in
                     the lightcone at lcDir, between minStep and maxStep, and numFiles=3,
                     then three text files will be written out, each containing 10 of the 
                     found halos. This option is intended to allow for separate cutout
                     runs being submitted in parallel, each handling a subset of all the
                     desired halos
    '''

    comm= MPI.COMM_WORLD
    rank = comm.Get_rank()
    numranks = comm.Get_size()
    if(rank==0):
        print('starting with {} MPI processes'.format(numranks))
        sys.stdout.flush()
    comm.Barrier()

    # get lightcone shells (step/snapshot numbers) matching those desired 
    # by minStep and maxStep
    lcHaloSubdirs = glob.glob('{}/*'.format(lcDir))
    
    # step number is assumed to be the last three chars of the subirectory names
    steps = np.array(sorted([int(s[-3:]) for s in lcHaloSubdirs]))
    steps = steps[np.logical_and(steps >= minStep, steps <= maxStep)]
    
    # hand out different steps to different ranks. randomize them at rank 0 to 
    # allow better load balancing
    np.random.seed(103)
    np.random.shuffle(steps)
    steps = np.array_split(steps, numranks)[rank]

    print("rank {} gets steps {}".format(rank, steps))
    sys.stdout.flush()
    comm.Barrier()
  
    # arrays to hold halos in the lc found above the desired massCut (to be written out)
    write_ids = np.array([])
    write_reps = np.array([])
    write_x = np.array([])
    write_y = np.array([])
    write_z = np.array([])
    write_redshift = np.array([])
    write_shells = np.array([])
    write_mass = np.array([])
    write_radius = np.array([])
    total=0

    # loop over lightcone shells
    for i in range(len(steps)):
        
        step = steps[i]
        if(step == 499): continue

        if(rank==0):
            print('\n---------- working on step {} ----------'.format(step))
            sys.stdout.flush()
       
        if(rank==0):
            print('reading lightcone')
            sys.stdout.flush()
        # there should only be one unhashed gio file in this subdir
        lc_file = sorted(glob.glob('{1}/*{0}/*'.format(step, lcDir)))[0]
        lc_tags = np.squeeze(gio.gio_read(lc_file, 'id'))
        lc_reps = np.squeeze(gio.gio_read(lc_file, 'replication')).astype(np.int32)
        lc_x = np.squeeze(gio.gio_read(lc_file, 'x'))
        lc_y = np.squeeze(gio.gio_read(lc_file, 'y'))
        lc_z = np.squeeze(gio.gio_read(lc_file, 'z'))
        lc_a = np.squeeze(gio.gio_read(lc_file, 'a'))
        
        # get halo redshifts and mask halo fof tags 
        # (the halo lightcone module outputs merger tree fof tags, including fragment bits)
        lc_redshift = 1/lc_a - 1
        lc_tags = (lc_tags * np.sign(lc_tags)) & 0x0000ffffffffffff 

        if(haloCat != None):
            
            if(massDef == 'fof'):
                cat_file = glob.glob('{1}/b0168/STEP{0}/*{0}*fofproperties'.format(step, haloCat))[0]
            elif(massDef == 'sod'):
                cat_file = glob.glob('{1}/M200/STEP{0}/*{0}*sodproperties'.format(step, haloCat))[0]
            else:
                raise Exception('Valid inputs for massDef are \'fof\', \'sod\'')

            if(rank==0):
                print('reading halo catalog at {}'.format(cat_file))
                sys.stdout.flush()
            fof_tags = np.squeeze(gio.gio_read(cat_file, 'fof_halo_tag'))
            halo_mass = np.squeeze(gio.gio_read(cat_file, '{}_halo_mass'.format(massDef)))
            if(massDef == 'sod'):
                halo_radius = np.squeeze(gio.gio_read(cat_file, 'sod_halo_radius'))
            else:
                halo_radius = np.zeros(len(halo_mass))

            if(rank==0):
                print('sorting')
                sys.stdout.flush()
            fof_sort = np.argsort(fof_tags)
            fof_tags = fof_tags[fof_sort]
            halo_mass = halo_mass[fof_sort]
            
            # Now we match to get the halo masses, with the matching done in the
            # following order:
            # lc masked fof_halo_tag > fof fof_halo_tag
            # fof fof_halo_tag > fof_halo_mass

            if(rank==0):
                print('matching lightcone to halo catalog to retrieve mass')
                sys.stdout.flush()
            lc_to_fof = None
            # fix this if ever want to use catalog mathcing feature ^
            #lc_to_fof = iv.search_sorted(fof_tags, lc_tags, sorter=np.argsort(fof_tags))

            # make sure that worked
            if(np.sum(lc_to_fof == -1) != 0):
                raise Exception('{0}% of lightcone halos not found in halo catalog. '\
                                'Maybe passed wrong files?'
                                .format(np.sum(lc_to_fof==-1)/float(len(lc_to_fof)) * 100))  

            lc_mass = halo_mass[lc_to_fof]
        
        else:
            lc_mass = np.squeeze(gio.gio_read(lc_file, '{}_mass'.format(massDef)))
            if(massDef == 'sod'):
                lc_radius = np.squeeze(gio.gio_read(lc_file, 'sod_radius'))
            else:
                lc_radius = None 
        if(rank == 0):
            print('read {} halos'.format(len(lc_mass)))
        
        # do mass cutting
        mass_mask = np.logical_and(lc_mass >= minMass, lc_mass < maxMass)
        lc_tags = lc_tags[mass_mask]
        lc_reps = lc_reps[mass_mask]
        lc_redshift = lc_redshift[mass_mask]
        lc_mass = lc_mass[mass_mask]
        lc_radius = lc_radius[mass_mask]
        lc_x = lc_x[mass_mask]
        lc_y = lc_y[mass_mask]
        lc_z = lc_z[mass_mask]

        # add these halos to write-out arrays
        if(rank==0):
            print('Found {0} halos ({1:.5f}% of all) within mass range of {2:.2e}-{3:.2e}'
                  .format(np.sum(mass_mask), (np.sum(mass_mask)/float(len(mass_mask)))*100, minMass, maxMass))
            sys.stdout.flush()
        if(rank==0):
            print('Appending halo data to write-out arrays')
            sys.stdout.flush()
        total += np.sum(mass_mask)
        if(rank==0):
            print('TOTAL: {}'.format(total))
            sys.stdout.flush()
        
        write_ids = np.hstack([write_ids, lc_tags])
        write_reps = np.hstack([write_reps, lc_reps]).astype(np.int32)
        write_x = np.hstack([write_x, lc_x]) 
        write_y = np.hstack([write_y, lc_y]) 
        write_z = np.hstack([write_z, lc_z]) 
        write_redshift = np.hstack([write_redshift, lc_redshift]) 
        write_shells = np.hstack([write_shells, np.ones(np.sum(mass_mask), dtype=np.int32)*step])
        write_mass = np.hstack([write_mass, lc_mass]) 
        write_radius = np.hstack([write_radius, lc_radius])
   
    # Do downsampling according to outFrac arg
    if(len(write_ids)) > 0:
        if(rank==0):
            print('\nDownsampling {0}% of {1} total halos'.format(outFrac*100, len(write_ids)))
            sys.stdout.flush()
        dsampling = np.random.choice(np.arange(len(write_ids)), int(len(write_ids)*outFrac), replace=False)
        write_ids = write_ids[dsampling]
        write_reps = write_reps[dsampling]
        write_x = write_x[dsampling]
        write_y = write_y[dsampling]
        write_z = write_z[dsampling]
        write_redshift = write_redshift[dsampling]
        write_shells = write_shells[dsampling]
        write_mass = write_mass[dsampling]
        write_radius = write_radius[dsampling]
    
    comm.Barrier()
    print('{} total halos found at rank {}'.format(len(write_ids), rank))
    sys.stdout.flush()
    comm.Barrier()

    # send all data to rank 0 for writing
    counts = None
    tot = None
    dspls = None
    numhalos = np.ones(1, dtype='i')*len(write_ids)
    if(rank == 0): 
        tot = np.zeros(1, dtype='i')
        counts = np.empty(numranks, dtype='i')
    comm.Reduce(numhalos, tot, op=MPI.SUM, root=0)
    comm.Gather(numhalos, counts, root=0)
    comm.Barrier()
    if(rank == 0):
        dspls = np.hstack([[0], np.cumsum(counts)[:-1]])
        print('{} total halos found across {} ranks'.format(tot, numranks))
        sys.stdout.flush()

    all_ids = None
    all_reps = None
    all_x = None
    all_y = None
    all_z = None
    all_redshift = None
    all_shell = None
    all_mass = None
    all_radius = None
    
    if(rank == 0):
        all_ids = np.empty(tot[0], dtype=np.int64)
        all_reps = np.empty(tot[0], dtype=np.int32)
        all_x = np.empty(tot[0], dtype=np.float64)
        all_y = np.empty(tot[0], dtype=np.float64)
        all_z = np.empty(tot[0], dtype=np.float64)
        all_redshift = np.empty(tot[0], dtype=np.float64)
        all_shell = np.empty(tot[0], dtype=np.int32)
        all_mass = np.empty(tot[0], dtype=np.float64)
        all_radius = np.empty(tot[0], dtype=np.float64)
        #print('Counts is {}, dspls is {}'.format(counts, dspls))
        print('Preparing to gather...')
    
    recv_ids = [all_ids, counts, dspls, MPI.LONG]
    recv_reps = [all_reps, counts, dspls, MPI.INT]
    recv_x = [all_x, counts, dspls, MPI.DOUBLE]
    recv_y = [all_y, counts, dspls, MPI.DOUBLE]
    recv_z = [all_z, counts, dspls, MPI.DOUBLE]
    recv_redshift = [all_redshift, counts, dspls, MPI.DOUBLE]
    recv_shell = [all_shell, counts, dspls, MPI.INT]
    recv_mass = [all_mass, counts, dspls, MPI.DOUBLE]
    recv_radius = [all_radius, counts, dspls, MPI.DOUBLE]

    comm.Gatherv([write_ids, numhalos], recv_ids, root=0)
    comm.Gatherv([write_reps, numhalos], recv_reps, root=0)
    comm.Gatherv([write_x, numhalos], recv_x, root=0)
    comm.Gatherv([write_y, numhalos], recv_y, root=0)
    comm.Gatherv([write_z, numhalos], recv_z, root=0)
    comm.Gatherv([write_redshift, numhalos], recv_redshift, root=0)
    comm.Gatherv([write_shell, numhalos], recv_shells, root=0)
    comm.Gatherv([write_mass, numhalos], recv_mass, root=0)
    comm.Gatherv([write_radius, numhalos], recv_radius, root=0)
    if(rank == 0):
        print('{} total halos gathered at rank 0'.format(len(all_ids)))
        sys.stdout.flush()
    

    # Now do writing to text file(s)
    if(rank==0):

        # remove duplicate halos
        print('Removing duplicate halos...')
        ask_unique = np.unique(all_ids, return_counts=True)
        unique_ids = ask_unique[0][ask_unique[1] == 1]
        id_mask = np.array([iid in unique_ids for iid in all_ids])

        # remove halos within correlation length of any replication boundary
        print('Removing halos with broken local correlations...')
        xm = np.abs((np.round(all_x/rL)*rL) - all_x) > corrLength
        ym = np.abs((np.round(all_y/rL)*rL) - all_y) > corrLength
        zm = np.abs((np.round(all_z/rL)*rL) - all_z) > corrLength
        pos_mask = np.logical_and(np.logical_and(xm, ym), zm)
        
        # and also any halos with a angular separation from the octant edge that 
        # is less than the max cutout angular scale phiMax 
        print('Removing halos within angular scale of octant boundary...')
        all_r = np.sqrt(all_x**2 + all_y**2 + all_z**2)
        phi = np.arctan(all_y/all_x)
        theta = np.arccos(all_z/all_r)
        phiMax = phiMax / 60 * np.pi/180.0
        thetaMax = phiMax
        ang_mask = np.logical_and(
                   np.logical_and(theta > thetaMax/2, theta < ((np.pi/2-thetaMax/2))), 
                   np.logical_and(phi > phiMax/2, phi < ((np.pi/2-phiMax/2))))
        
        # combine id, spatial, and anuglar masks from above and cut
        mask = np.logical_and(np.logical_and(pos_mask, ang_mask), id_mask)
        all_ids = all_ids[mask]
        all_reps = all_reps[mask]
        all_x = all_x[mask]
        all_y = all_y[mask]
        all_z = all_z[mask]
        all_redshift = all_redshift[mask]
        all_shell = all_shell[mask]
        all_mass = all_mass[mask]
        all_radius = all_radius[mask]
        print('Removed {} total halos ({} spatial rejections and {} duplicates)'.format(np.sum(~mask), 
               np.sum(np.logical_and(~pos_mask, ~ang_mask)), np.sum(~id_mask)))

        print('\nDone, obtained {0} total halos to write across {1} text files'
              .format(len(all_ids), numFiles))
        sys.stdout.flush()
        
        # and combine tags and replication identifiers for unqiue ids
        combine_tags_reps = np.frompyfunc("{}_{}".format, 2, 1)
        all_ids = combine_tags_reps(all_ids, all_reps)
        
        write_masks = np.array_split(np.arange(len(all_ids), dtype='i'), numFiles)
        for j in range(numFiles):
            #READY TO WRITE SHELLS; NEED TO UPDATE OLD FILES 
            wm = write_masks[j]
            next_file = open("{0}/lcHaloList_mass{1:.2e}-{2:.2e}_steps{3}-{4}_{5}.txt"
                             .format(outDir, minMass, maxMass, minStep, maxStep, j), 'w')
            print('writing {0} halos to file {1}'.format(len(wm), j+1))
            sys.stdout.flush()
            
            for n in range(len(wm)):
                if(massDef == 'sod'):
                    next_file.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(
                                                               all_ids[wm][n], all_redshift[wm][n], 
                                                               all_shell[wm][n], all_mass[wm][n], all_radius[wm][n], 
                                                               all_x[wm][n], all_y[wm][n], all_z[wm][n]))
                elif(massDef == 'fof'):
                    next_file.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(
                                                               all_ids[wm][n], all_redshift[wm][n], 
                                                               all_shell[wm][n], all_mass[wm][n], 
                                                               all_x[wm][n], all_y[wm][n], all_z[wm][n]))
            next_file.close()
        print('Done')
        sys.stdout.flush()
        

# =================================================================================================


def vis_output_regions(maxStep, minStep, rL, corrLength, phiMax, plotZ=False, sliceZ=False, outDir=None):

    '''
    Function to visualize the spatial distribution of the output of list_halos()
    given the same input parameters. In a 2d projection, a plot will be rendered
    which shows the rejected halo populations at the replicated box boundaries, 
    and within an angular limit with respect to the octant edge

    Params:
    :param maxStep: The maximum simulation snapshot step to include
    :param minStep: The minimum simulation snapshot step to include
    :param rL: The length of the box underlying the input lightcone in Mpc/h
    :param corrLength: The length scale within which to consider correlated 
                       with a halo-- halos within this length of a replicated box
                       boundary will experience correlation breaks in the density field
                       and will be rejected
    :param phiMax: the maximum angular scale that one may want to use on subsequent
                   halo lightcone cutouts given the halos output by list_halos(), 
                   in arcseconds
    :param plotZ: whether or not to solve the problem in 3d including the z-axis 
                  If False, a 2d plot is given which isn't exactly a projection 
                  (does not correspond to the x-y plane at any particular z-position), 
                  but rather demonstates the 2 dimensional anlog of the problem
    :param sliceZ: if plotZ is True, then this argument controls the plotting type. If
                   False, then a 3d plot is displayed. If True, then 15 projected z-slices
                   are shown
    :param outDir: Where to save plots if plotZ = sliceZ = True
    :return: None
    '''

    from astropy.cosmology import WMAP7 as cosmo
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import rc

    a = np.linspace(1/201, 1, 500)
    z = 1/a-1
    zRange = z[np.array([minStep, maxStep])]
    LRange = cosmo.comoving_distance(zRange).value * cosmo.h
    
    x = np.random.rand(50000)*LRange[0]
    y = np.random.rand(50000)*LRange[0]
    z = np.random.rand(50000)*LRange[0]
    if(plotZ): r = np.sqrt(x**2 + y**2 + z**2)
    else: r = np.sqrt(x**2 + y**2)
    
    lc_mask = r < LRange[0]
    x = x[lc_mask]
    y = y[lc_mask]
    z = z[lc_mask]
    r = r[lc_mask]
    
    phiMax = phiMax / 60 * np.pi/180.0
    phi = np.arctan(y/x)
    if(plotZ):
        theta = np.arccos(z/r)
        thetaMax = phiMax
    maxRep = max([np.max((x/rL).astype(int))+1, np.max((y/rL).astype(int))+1, np.max((z/rL).astype(int))+1])

    # find all points which lie within the correlation length of any replication boundary
    # and also any points which have an angualr separation from the octant edge less than phiMax
    xm = np.abs((np.round(x/rL)*rL) - x) > corrLength
    ym = np.abs((np.round(y/rL)*rL) - y) > corrLength
    zm = np.abs((np.round(z/rL)*rL) - z) > corrLength
    if(plotZ): 
        pm = np.logical_and(np.logical_and(xm, ym), zm)
        tm = ttm = np.logical_and(
                   np.logical_and(theta > thetaMax/2, theta < ((np.pi/2-thetaMax/2))), 
                   np.logical_and(phi > phiMax/2, phi < ((np.pi/2-phiMax/2)))) 
    else: 
        pm = np.logical_and(xm, ym)
        tm = np.logical_and(phi > phiMax/2, phi < ((np.pi/2-phiMax/2))) 
    mm = np.logical_and(pm, tm)


    rc('text', usetex=True)

    # 3d plot
    if(plotZ and not sliceZ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x[mm], y[mm], z[mm], '.c', lw=0, ms=2)
        ax.plot(x[~mm], y[~mm], z[~mm], '.r', lw=0, ms=2)
        ax.set_zlabel(r'$z\>[Mpc/h]$', fontsize=16)
        plt.plot(xx, yy, zz, '.m', ms=4)
    
    # 2d slices in z
    elif(plotZ and sliceZ):
        zOrd = np.argsort(z)
        xsl = np.array_split(x[zOrd], 15)
        ysl = np.array_split(y[zOrd], 15)
        zsl = np.array_split(z[zOrd], 15)
        mmsl = np.array_split(mm[zOrd], 15)
        for i in range(15):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xsl[i][mmsl[i]], ysl[i][mmsl[i]], '.c', lw=0, ms=2)
            ax.plot(xsl[i][~mmsl[i]], ysl[i][~mmsl[i]], '.r', lw=0, ms=2)
            ax.set_xlabel(r'$x\>[Mpc/h]$', fontsize=16)
            ax.set_ylabel(r'$y\>[Mpc/h]$', fontsize=16)
            ax.set_title(r"Valid Cutout Regions" "\n" r"$\phi_\mathrm{max}=1000\mathrm{arcsec},\>l_\xi=L/10$")
            # plot box replication edges
            for j in range(maxRep):
                ax.plot([j*rL, (j+1)*rL], [j*rL, j*rL], lw=2, color='k')
                ax.plot([j*rL, (j+1)*rL], [(j+1)*rL, (j+1)*rL], lw=2, color='k')
                ax.plot([j*rL, j*rL], [j*rL, (j+1)*rL], lw=2, color='k')
                ax.plot([(j+1)*rL, (j+1)*rL], [j*rL, (j+1)*rL], lw=2, color='k')
            ax.set_xlim([np.min(x), np.max(x)])
            ax.set_ylim([np.min(y), np.max(y)])
            plt.show()
            plt.savefig('{}/{}.png'.format(outDir, i))
        return
    
    # 2d problem ignoring z
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x[mm], y[mm], '.c', lw=0, ms=2)
        ax.plot(x[~mm], y[~mm], '.r', lw=0, ms=1)        
        # plot box replication edges
        for j in range(maxRep):
            ax.plot([j*rL, (j+1)*rL], [j*rL, j*rL], lw=2, color='k')
            ax.plot([j*rL, (j+1)*rL], [(j+1)*rL, (j+1)*rL], lw=2, color='k')
            ax.plot([j*rL, j*rL], [j*rL, (j+1)*rL], lw=2, color='k')
            ax.plot([(j+1)*rL, (j+1)*rL], [j*rL, (j+1)*rL], lw=2, color='k')
        ax.set_xlim([np.min(x), np.max(x)])
        ax.set_ylim([np.min(y), np.max(y)])
    
    ax.set_xlabel(r'$x\>[Mpc/h]$', fontsize=16)
    ax.set_ylabel(r'$y\>[Mpc/h]$', fontsize=16)
    ax.set_title(r"Valid Cutout Regions" "\n" r"$\phi_\mathrm{max}=1000\mathrm{arcsec},\>l_\xi=L/10$")
    plt.show()


# =================================================================================================


def list_alphaQ_halos(maxStep=499, minStep=247, minMass=10**13.5, maxMass=1e18, outFrac=0.01, numFiles=1):
    
    '''
    This function runs list_halos with data paths predefined for AlphaQ.
    Function parameters are as given in the docstrings above for list_halos
    '''

    list_halos(lcDir='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_halos',
               haloCat='/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Halos/M200',
               outDir='/home/hollowed/cutout_run_dirs/alphaQ/cutout_alphaQ_full',
               massDef = 'sod', rL = 256, corrLength=150, phiMax=1000, 
               maxStep=maxStep, minStep=minStep, minMass=minMass, maxMass=maxMass, 
               outFrac=outFrac, numFiles=numFiles)


def list_outerRim_halos(maxStep=499, minStep=121, minMass=10**13.5, maxMass=1e18, outFrac=0.01, numFiles=1):
    
    '''
    This function runs list_halos with data paths predefined for OuterRim.
    Function parameters are as given in the docstrings above for list_halos
    '''

    list_halos(lcDir='/projects/DarkUniverse_esp/jphollowed/outerRim/lightcone_halos_octant_matchup',
               outDir='/home/hollowed/cutout_run_dirs/outerRim/cutout_outerRim_full',
               haloCat = None, massDef = 'sod', rL=3000, corrLength=150, phiMax=1000, 
               maxStep=maxStep, minStep=minStep, minMass=minMass, maxMass=maxMass, 
               outFrac=outFrac, numFiles=numFiles)

if __name__ == "__main__":
    minMass = 10**float(sys.argv[1])
    maxMass = 10**float(sys.argv[2])
    outFrac = float(sys.argv[3])
    numFiles = int(sys.argv[4])
    maxStep = int(sys.argv[5])
    minStep = int(sys.argv[6])
    list_outerRim_halos(maxStep=maxStep, minStep=minStep, minMass=minMass, maxMass=maxMass, outFrac=outFrac, numFiles=numFiles)
