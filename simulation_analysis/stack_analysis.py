
'''
Joe Hollowed
Last edited 8/2/2017

A collection of functions to do some analysis on the stacked halos output from 
stack_cores.py
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import os
import pdb
import math
import glob
import h5py
import coreTools
import numpy as np
import dispersionStats as stat
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rcfn


#==============================================================================================

def stack_segregation(n_draws=21, max_r=9, max_z=1.5, catalog=0, velComp =2, norm=0, 
                      processed = True):
    '''
    Caclulate the segregation in velocity dispersion of cores on each core property (i.e 
    infall_mass) by drawing from the full stacked halo ("full" as in, across all redshifts, 
    up to max_z). 
    
    As an example, assume this segregation is being preformed on only the property
    infall_mass, otherwie with the default parameters:
    This is done just as the approach described in Bayliss et al. 2016, where we asign some mass 
    cut defining the boundary between "high infall mass" and "low infall mass". Then, we draw cores
    from the full-stack (up to redshift 1 and radial distance r/r200 = 1), and vary the 
    fraction of cores that we draw from the "high" and "low" infall_mass populations. We then 
    calculate the velocity dispersion of each "mixed mass" bin, thus learning about the 
    "segregation". The first bin, then, will be 100% populated by low infall_mass cores, the 
    center bin populated by 50% low and 50% high infall_mass cores, and the last bin populated 
    100% by high infall_mass cores. We call these "mixed bins"

    This entire procedure is done using both 1D and 3D velocities/dispersions, and for all 
    important core properties (infall mass, radius, infall time)

    :param n_draws: the number of mixed bins to include in the analysis (default=21)
    :param max_r: maximum normalized radial distance of which to draw cores from (default=1)
    :param max_z: the maximum redshift from which to draw cores
    :param catalog: version of core velocities to use (0=biweight, 1=median, 2=central particle)
    :param velComp: which velocity component to use (0=radial, 1=tangent, or 2=total velocity vector)
    :param norm: which velocity normalization to use (0=coreNorm, 1=haloNorm)
    :param processed: whether or not to use the processed core catalog
    
    :return: None
    '''

    # find stacked-halo core data and prepare parameters
    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    processPrefix = ['un', ''][processed]
    corePath = 'data/coreCatalog/{}'.format(catalogName)
    stackFile = h5py.File('{}/stackedCores_{}processed.hdf5'.format(corePath, processPrefix), 'r+')
    velComp = ['_rad', '_tan', ''][velComp]
    norm = ['_coreNorm', '_haloNorm'][norm]

    # build suffix for the tail of the filename based on the current parameters
    if(velComp == ''): suff = 'z{}r{}n{}'.format(max_z, max_r, norm[1])
    else: suff = 'z{}r{}n{}v{}'.format(max_z, max_r, norm[1], velComp[1])
    
    # create new hdf5 file to hold segregation data
    segrStackFile = h5py.File('{}/segregatedCores_{}processed_{}.hdf5'
                              .format(corePath, processPrefix, suff), 'w')
    print('\nRead data from core catalog and created file for segregation')
    print('Using {} velocity with {} normalization'.format(catalogName, norm))

    # find all cores within redshift bound set by max_z, from stacked halo
    steps = np.array(list(stackFile.keys()))
    zs = np.array([stackFile[step].attrs['z'] for step in steps])
    zMask = (zs <= max_z)
    
    stepNums = np.array([int(step.split('_')[-1]) for step in steps[zMask]])
    sortMask = np.argsort(stepNums)
    
    stepNums = stepNums[sortMask]
    zs = zs[zMask][sortMask]
    zStacks = [stackFile[step] for step in steps[zMask][sortMask]]

    # list each core property to be segregated
    segrProp = ['infall_mass', 'radius', 'time_since_infall']
    
    # record current parameters as hdf root group attributes, and
    # create a new hdf group for each mixed bin
    segrStackFile.attrs.create('max_(r/r200)', max_r)
    segrStackFile.attrs.create('max_z', max_z)
    segrStackFile.attrs.create('norm', norm.encode('utf8'))
    for j in range(len(stepNums)):
        nextStep = segrStackFile.create_group('step_{}'.format(stepNums[j]))
        nextStep.attrs.create('z', zs[j])
        for prop in segrProp:
            nextStep.create_group('{}_segr'.format(prop)) 

    #---------------------------------------------------------------------------------
    #-------------------- begin segregating stacks at each redshift -------------------
    
    for n  in range(len(zStacks)):

        # focus on hdf group from the original stacked halo that corresponds to the current redshift
        # Note: h5py datasets must be read in the form file['datasetName'][:] to yield 
        # an actual numpy array, file['datasetName'] will return an h5py object
        step = stepNums[n]
        stack = zStacks[n]
        rMask = stack['r_norm'][:] <= max_r
        z = zs[n]
        print('\n\n----- Working on stack at z = {} ({}/{})-----'.format(z, n, len(zStacks))) 

        # get dispersion and center of the full stack, before segregation
        # Note: it is important to draw without replacement when bootstrapping if you are investigating 
        # a second order statistic, like dispersion, since allowing replacement will artificially 
        # bias results low (Bayliss + 2016), hence we pass repl=False to the bootstrap function
        velocities = stack['v{}{}'.format(velComp, norm)][:][rMask]
        [vDisp_all, vDisp_allErr] = stat.bootstrap_bDispersion(velocities, 
                                                               size=len(velocities)/2, repl = False)
        vAvg_all = stat.bAverage(velocities)
        
        # create vector of fractions according to n_draws
        fracs = np.linspace(0, 1, n_draws)
        bin_dispersions = np.zeros(n_draws)
        bin_disp_errors = np.zeros(n_draws)
        
        # add a a new column of binary data for each segregation property to the original
        # stacked cluster hdf file, representing if that core belong to the lower or upper pop
        for prop in segrProp:

            print('---------- preforming segregation on {} ----------'.format(prop))

            # skip this if it's already been done for the current stack file
            if('{}_segr'.format(prop) not in list(stack.keys())):
                # segregate cores into low or high distribution on a halo-by-halo basis
                halo_tags = np.unique(stack['fof_halo_tag'][:])
                nCores = len(stack['core_tag'][:])
                popLabels = np.zeros(nCores) 
                
                for halo_tag in halo_tags:
                    haloMask = np.where(stack['fof_halo_tag'][:] == halo_tag)
                    segr_data = stack[prop][:][haloMask]
                    prop_threshold = np.median(segr_data)
                    halo_upperPopulation = (segr_data > prop_threshold)
                    popLabels[haloMask] = halo_upperPopulation
                
                stack.create_dataset('{}_segr'.format(prop), data = popLabels.astype(int))
            else:
                print('skipping segregation on {}; already complete'.format(prop))

            for j in range(n_draws):
                print('Working on bin {} ({:.2f}% upper points)'.format(j, fracs[j]*100))
                
                # the number of cores used in the segregation analysis, 
                # N_tot should not exceed the minimum of these two populations
                lower_sample = np.where(stack['{}_segr'.format(prop)][:] == 0)[0]
                upper_sample = np.where(stack['{}_segr'.format(prop)][:] == 1)[0]
                N_tot = min(len(lower_sample), len(upper_sample))
                
                # use fractions to compile a mixed mass sample for this bin
                lower_pop = math.ceil(N_tot*(1-fracs[j]))
                upper_pop = math.floor(N_tot*fracs[j])
                sample = np.concatenate([lower_sample[0:lower_pop], upper_sample[0:upper_pop]])
                sample_v = stack['v{}{}'.format(velComp, norm)][:][sample]

                # find the dispersion of the cores in this bin, and associated errors, via 
                # bootstrap resampling 
                [vDisp_avg, vDisp_err] = stat.bootstrap_bDispersion(sample_v, 
                                                                    size=len(sample_v)/2, repl = False)

                # normalize the resultant dispersions and errors by the dispersion of the 
                # entire stacked halo (using proper error propegation for the uncertainty)
                bin_dispersions[j] = vDisp_avg / vDisp_all
                bin_disp_errors[j] = bin_dispersions[j] * np.sqrt(vDisp_err**2 + vDisp_allErr**2)

                print('Done. dispersion = {} +- {}'.format(bin_dispersions[j], bin_disp_errors[j]))
            
            print('saving segregation data')    
            dataFile = segrStackFile['step_{}'.format(step)]['{}_segr'.format(prop)]
            dataFile.create_dataset('pcen_lowerPop', data = fracs)
            dataFile.create_dataset('bin_vDisp', data = bin_dispersions)
            dataFile.create_dataset('bin_vDispErr', data = bin_disp_errors)

            print('Done.')    


#==============================================================================================

def segregation_binning(n_bins=21, max_r=9, max_z=1.5, catalog=0, velComp =2, norm=0, 
                        processed = True):
    '''
    Similar the the stack_segregation() function, where I simply bin the stacked halos by
    some core property rather than segregating by it (i.e. instead of collecting two distinct
    populations and drawing from each in varying percentage, I just collect n_bins distinct
    populations, and measure the dispersion of each.

    I still use the same bootstrap method for finding the dispersion and dispersion error
    for each distribution (bin of cores). This can also all be done using 1D or 3D velocities.

    :param n_bins: the number of mixed bins to include in the analysis (default=21)
    :param max_r: maximum normalized radial distance of which to draw cores from (default=1)
    :param max_z: the maximum redshift from which to draw cores
    :param catalog: version of core velocities to use (0=biweight, 1=median, 2=central particle)
    :param velComp: which velocity component to use (0=radial, 1=tangent, or 2=total velocity vector)
    :param norm: which velocity normalization to use (0=coreNorm, 1=haloNorm)
    :param processed: whether or not to use the processed core catalog

    :return: None, save results to hdf file
    '''

    # find stacked-halo core data and prepare parameters
    catalogName = ['BLEVelocity', 'MedianVelocity', 'CentralVelocity'][catalog]
    processPrefix = ['un', ''][processed]
    corePath = 'data/coreCatalog/{}'.format(catalogName)
    stackFile = h5py.File('{}/stackedCores_{}processed.hdf5'.format(corePath, processPrefix), 'r')
    velComp = ['_rad', '_tan', ''][velComp]
    norm = ['_coreNorm', '_haloNorm'][norm]

    # build suffix for the tail of the filename based on the current parameters
    if(velComp == ''): suff = 'z{}r{}n{}'.format(max_z, max_r, norm[1])
    else: suff = 'z{}r{}n{}v{}'.format(max_z, max_r, norm[1], velComp[1])
    
    # create new hdf5 file to hold segregation data
    segrStackFile = h5py.File('{}/segregatedCores_binned_{}processed_{}.hdf5'
                              .format(corePath, processPrefix, suff), 'w')
    print('\nRead data from core catalog and created file for segregation')
    print('Using {} velocity with {} normalization'.format(catalogName, norm))

    # find all cores within redshift bound set by max_z, from stacked halo
    steps = np.array(list(stackFile.keys()))
    zs = np.array([stackFile[step].attrs['z'] for step in steps])
    zMask = (zs <= max_z)
    
    stepNums = np.array([int(step.split('_')[-1]) for step in steps[zMask]])
    sortMask = np.argsort(stepNums)
    
    stepNums = stepNums[sortMask]
    zs = zs[zMask][sortMask]
    zStacks = [stackFile[step] for step in steps[zMask][sortMask]]

    # list each core property to be segregated
    segrProp = ['infall_mass', 'radius', 'time_since_infall']
    
    # record current parameters as hdf root group attributes, and
    # create a new hdf group for each mixed bin
    segrStackFile.attrs.create('max_(r/r200)', max_r)
    segrStackFile.attrs.create('max_z', max_z)
    segrStackFile.attrs.create('norm', norm.encode('utf8'))
    for j in range(len(stepNums)):
        nextStep = segrStackFile.create_group('step_{}'.format(stepNums[j]))
        nextStep.attrs.create('z', zs[j])
        for prop in segrProp:
            nextStep.create_group('{}_segr'.format(prop)) 

    #---------------------------------------------------------------------------------
    #-------------------- begin segregating stacks at each redshift -------------------
    
    for n  in range(len(zStacks)):

        # focus on hdf group from the original stacked halo that corresponds to the current redshift
        # Note: h5py datasets must be read in the form file['datasetName'][:] to yield 
        # an actual numpy array, file['datasetName'] will return an h5py object
        step = stepNums[n]
        stack = zStacks[n]
        rMask = stack['r_norm'][:] <= max_r
        z = zs[n]
        print('\n\n----- Working on stack at z = {} ({}/{})-----'.format(z, n, len(zStacks))) 

        # get dispersion and center of the full stack, before segregation
        # Note: it is important to draw without replacement when bootstrapping if you are investigating 
        # a second order statistic, like dispersion, since allowing replacement will artificially 
        # bias results low (Bayliss + 2016), hence we pass repl=False to the bootstrap function
        velocities = stack['v{}{}'.format(velComp, norm)][:][rMask]
        [vDisp_all, vDisp_allErr] = stat.bootstrap_bDispersion(velocities, 
                                                               size=len(velocities)/2, repl = False)
        vAvg_all = stat.bAverage(velocities)
        
        # add a a new column of binary data for each segregation property to the original
        # stacked cluster hdf file, representing if that core belong to the lower or upper pop
        for prop in segrProp:

            print('---------- preforming segregation on {} ----------'.format(prop))
            
            # bin stack into bins of equal sample size on the current segregation property
            propData = stack[prop][:]
            bin_edges = np.percentile(propData, np.linspace(0, 100, n_bins))
            bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
            bin_widths = np.diff(bin_edges)
            
            bin_dispersions = np.zeros(n_draws)
            bin_disp_errors = np.zeros(n_draws)

            for j in range(n_bins):
                print('Working on bin {} ({:.2f}% upper points)'.format(j, fracs[j]*100))
                
                bin_mask = (propData > bin_edges[j]) & (prop <= bin_edges[j+1])
                bin_vs = stack['v{}{}'.format(velComp, norm)][:][binMask]
                
                # find the dispersion of the cores in this bin, and associated errors, via 
                # bootstrap resampling 
                [vDisp_avg, vDisp_err] = stat.bootstrap_bDispersion(bin_vs, 
                                                                    size=len(bin_vs)/2, repl = False)

                # normalize the resultant dispersions and errors by the dispersion of the 
                # entire stacked halo (using proper error propegation for the uncertainty)
                bin_dispersions[j] = vDisp_avg / vDisp_all
                bin_disp_errors[j] = bin_dispersions[j] * np.sqrt(vDisp_err**2 + vDisp_allErr**2)

                print('Done. dispersion = {} +- {}'.format(bin_dispersions[j], bin_disp_errors[j]))
            
            print('saving segregation data')    
            dataFile = segrStackFile['step_{}'.format(step)]['{}_segr'.format(prop)]
            dataFile.create_dataset('bin_center', data = bin_centers)
            dataFile.create_dataset('bin_width', data = bin_widths)
            dataFile.create_dataset('bin_vDisp', data = bin_dispersions)
            dataFile.create_dataset('bin_vDispErr', data = bin_disp_errors)

            print('Done.')    


#==============================================================================================

def V_vs_R(max_r = 2.5, dim=3, plot=False):
    '''
    Calculate and plot the average core velocity as a function of radius in
    stacked halos. The procedure is to bin the core data by radius, find the mean
    core velocity and uncertainty in each bin, and plot the results. 
    
    The width of the radius bins are determined such that they all hold the same
    number of data points (by taking percentiles of the core data). The number of
    bins is decided rather randomly, with the aim to hav enough bins to represent the data, 
    without having too few that a very low number of cores lands in each bin. A number
    of bins that seems to satisfy this is 2/3 the number of halos present in the working 
    stacked halo.

    :param max_r: the maximum radius (in units of r200) to cut off the analysis; going too far
              can include far outlying cores from overlinked halos. (default = 2.5)
    :param dim: dimensions with which to preform the analysis. This simply means that if dim=3,
            use radial velocities and 3D dispersions. If dim=1, use 1d velocities and 
            dispersions. Valid values are 1 and 3. (default = 3)
    :param plot: whether or not to plot the results (default=False)
    :return: None        
    '''

    if(dim != 1 and dim != 3): raise ValueError('arg \'dim\' must be 1 or 3')

    path = '/home/jphollowed/data/hacc/alphaQ/coreCatalog_merged/stackedHalos/by_redshift'
    save_dest = '/home/jphollowed/data/hacc/alphaQ/coreCatalog_merged/stackAnalysis/VvsR_cores'
    if(dim==3):fig_path = '/home/jphollowed/figs/VvsR_figs/merged_figs'
    elif(dim==1):fig_path='/home/jphollowed/figs/VvsR_1D_figs/merged_figs'
    stack_files = sorted(glob.glob('{}/stack*'.format(path)))

    for f in stack_files:
        
        stack = np.load(f)
        N_halo = int(f.split('_')[-1].split('h')[0])
        N_core = len(stack)
        z = f.split('_')[-2]
        print('\nworking on stack at z = {}'.format(z))

        if(dim == 3):
            r = stack['r_rad_norm']
            rMask = r < max_r
            r = r[rMask]
            v_c = stack['v_rad_coreNorm'][rMask]
            v_d = stack['v_rad_dmNorm'][rMask]
        if(dim == 1):
            r = stack['r_rad_2d_norm']
            rMask = r < max_r
            r = r[rMask]
            v_c = stack['v_1d_coreNorm'][rMask]
            v_d = None

        n_bins = N_halo * 0.66
        bin_edges = np.percentile(r, np.linspace(0, 100, n_bins).tolist())
        avg_r = [(bin_edges[n+1] + bin_edges[n]) / 2 for n in range(len(bin_edges)-1)]
        avg_v = np.empty(len(avg_r))
        err = np.zeros(len(avg_r))

        for i in range(len(avg_v)):
            if(i%10==0): print('working on bin {}'.format(i))
            mask = (r <= bin_edges[i+1]) & (r > bin_edges[i])
            vs = v_c[mask]
            bootstrap_means = np.zeros(1000)

            for j in range(len(bootstrap_means)): 
                next_sample= np.random.choice(vs, size=len(vs), replace=True)
                bootstrap_means[j] = np.mean(next_sample)
            
            avg_v[i] = np.mean(vs)
            err[i] = np.std(bootstrap_means)
        
        cols=['avg_r', 'avg_v', 'error']
        stack_output = np.rec.fromarrays([avg_r, avg_v, err], names=cols)
        if(dim==3): np.save('{}/VvsR_cores_{}.npy'.format(save_dest, z), stack_output)
        if(dim==1): np.save('{}/VvsR_1D_cores_{}.npy'.format(save_dest, z), stack_output)
        
        if(plot):
    
            plt.rc('text', usetex=True)
            params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
            plt.rcParams.update(params)
            plt.rcParams['mathtext.fontset'] = 'custom'
            plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
            plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
            plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            if(dim==3):    
                ax1.plot(r, v_c, 'x', label='radial core velocity', color=[.7, .7, 1])
                ax1.plot(avg_r, avg_v, '+r', ms=10, mew=1.5, label='bin avg velocity')
                ax1.fill_between(avg_r, avg_v-err, avg_v+err, color=[1, .7, .7])
                ax1.plot([0, 2.5], [0, 0], '--', color='black', linewidth=1.5)
                ax1.set_xlabel(r'$r/r_{200}$', fontsize=24)
                ax1.set_ylabel(r'$v_{radial}/\sigma_{v(cores),3D}$', fontsize=24)
            elif(dim==1):
                ax1.plot(r, v_c, 'x', label='projected core velocity', color=[.7, .7, 1])
                ax1.plot(avg_r, avg_v, '+r', ms=10, mew=1.5, label='bin avg velocity')
                ax1.fill_between(avg_r, avg_v-err, avg_v+err, color=[1, .7, .7])
                ax1.plot([0, 2.5], [0, 0], '--', color='black', linewidth=1.5)
                ax1.set_xlabel(r'$r_{2D}/r_{200}$', fontsize=24)
                ax1.set_ylabel(r'$v_{1D}/\sigma_{v(cores),1D}$', fontsize=24)
            ax1.set_ylim([-4, 4])
            ax1.set_xlim([0, 2.5])
            ax1.legend()

            ax1.text(0.05, 0.85, 'z = {}\n{} halos\n{} total cores'.format(z, N_halo, N_core), 
                 transform=ax1.transAxes, fontsize = 16)

            plt.savefig('{}/{}.png'.format(fig_path, z))
            print('saved figure for z = {}'.format(z))
