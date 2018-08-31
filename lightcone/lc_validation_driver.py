'''
This file runs all of the validation test written in lc_interpolation_validation with ease
'''

import lc_interpolation_validation as vt
import pdb

def matchingTests(lcDir_interp, lcDir_extrap, snapshotDir, figDir, trajectoryDataDir, trajectoryPlotDir, 
                  duplicateDataDir, pmode, snapSubdirs, plotMode, smode, cmap):
    
    allSteps = [401, 411, 421, 432, 442, 453, 464, 475, 487, 499]
    
    # ----------------------------------------------------------------------
    
    print('\n\n ==== Histograms validation test =====')
    vt.lightconeHistograms(lcDir_interp, lcDir_extrap, step=442, rL=256, 
                           plotMode=plotMode, outDir=figDir)
    
    # ----------------------------------------------------------------------

    print('\n\n ==== Particle trajectory validation test =====')
    diffRanges = ['min', 'med', 'max']
    for diff in diffRanges:
        vt.saveLightconePathData(lcDir_extrap, lcDir_interp, snapshotDir, trajectoryDataDir,
                                 rL=256, diffRange=diff, snapshotSubdirs=snapSubdirs, 
                                 mode=pmode, solverMode=smode)

    # ----------------------------------------------------------------------
    
    print('\n\n ==== Plotting particle trajectory validation test =====')
    diffRanges = ['min', 'med', 'max']
    for diff in diffRanges:
        vt.plotLightconePaths(trajectoryDataDir, diffRange=diff, plotMode=plotMode, outDir=trajectoryPlotDir)
    
    # ----------------------------------------------------------------------
    
    print('\n\n ==== Finding duplicates validation test =====')
    if(pmode == 'halos'):
        vt.findDuplicates(lcDir_interp, steps=[432, 442, 453], 
                          lcSuffix='interp', outDir=duplicateDataDir,
                          mode=pmode, mergerTreeDir=snapshotDir, solverMode=smode)
        vt.findDuplicates(lcDir_extrap, steps=[432, 442, 453], 
                          lcSuffix='extrap', outDir=duplicateDataDir,
                          mode=pmode, mergerTreeDir=snapshotDir, solverMode=smode)
    else:
        vt.findDuplicates(lcDir_interp, steps=[432, 442], 
                          lcSuffix='interp', outDir=duplicateDataDir,
                          mode=pmode, initialVolumeOnly=False, solverMode=smode)
        vt.findDuplicates(lcDir_extrap, steps=[432, 442], 
                          lcSuffix='extrap', outDir=duplicateDataDir,
                          mode=pmode, initialVolumeOnly=False, solverMode=smode) 
    
    # ----------------------------------------------------------------------
    
    print('\n\n ==== Plotting duplicates scatter validation test =====')
    vt.compareDuplicates(duplicateDataDir, steps=[432, 442], lcSuffix=['interp', 'extrap'],
                         plotMode=plotMode, outDir=figDir)


# ==================================================================================================
# ==================================================================================================


def comparisonTests(lcDir_interp, lcDir_extrap, snapshotDir, fofDir, figDir, duplicateDataDir, pairwiseDataDir, 
                    pmode, snapSubdirs, plotMode, smode, cmap):


    #allSteps = [401, 411, 421, 432, 442, 453, 464, 475, 487, 499]
    allSteps = [180, 184, 189, 198, 203, 208, 213, 219, 224]

    
    if(0):
        print('\n\n ==== Finding duplicates validation test =====')
        if(pmode == 'halos'):
            for k in range(len(allSteps)-2):
                vt.findDuplicates(lcDir_interp, steps=[allSteps[k], allSteps[k+1], allSteps[k+2]], 
                                  lcSuffix='interp', outDir=duplicateDataDir,
                                  mode=pmode, mergerTreeDir=snapshotDir, solverMode='backward')
                vt.findDuplicates(lcDir_extrap, steps=[allSteps[k], allSteps[k+1], allSteps[k+2]], 
                                  lcSuffix='extrap', outDir=duplicateDataDir,
                                  mode=pmode, mergerTreeDir=snapshotDir, solverMode='forward')
            vt.findDuplicates(lcDir_interp, steps=[487, 499], 
                              lcSuffix='interp', outDir=duplicateDataDir,
                              mode=pmode, mergerTreeDir=snapshotDir, solverMode='backward')
            vt.findDuplicates(lcDir_extrap, steps=[487, 499], 
                              lcSuffix='extrap', outDir=duplicateDataDir,
                              mode=pmode, mergerTreeDir=snapshotDir, solverMode='forward')
        
        else:
            for k in range(len(allSteps)-1):
                vt.findDuplicates(lcDir_interp, steps=[allSteps[k], allSteps[k+1]], 
                                  lcSuffix='interp', outDir=duplicateDataDir,
                                  mode=pmode, initialVolumeOnly=False)
                vt.findDuplicates(lcDir_extrap, steps=[allSteps[k], allSteps[k+1]], 
                                  lcSuffix='extrap', outDir=duplicateDataDir,
                                  mode=pmode, initialVolumeOnly=False) 
 
        # ----------------------------------------------------------------------
        
        print('\n\n ==== Plotting duplicates histogram validation test =====')
        vt.compareDuplicates(duplicateDataDir, steps=allSteps, 
                             lcSuffix=['interp', 'extrap'],
                             plotMode=plotMode, outDir=figDir)

        # ----------------------------------------------------------------------
        print('\n\n ==== N(z) validation test =====')
        plotMode='show'
        vt.N_z([lcDir_interp], steps=allSteps[::-1], 
                         plotMode=plotMode, outDir = figDir)
        
    # ----------------------------------------------------------------------
    
    print('\n\n ==== Compare Replications validation test =====')
    vt.compareReps(lcDir_interp, lcDir_extrap, step=213, plotMode=plotMode, outDir=figDir)

    # ----------------------------------------------------------------------
    
    print('\n\n ==== Comv Dist vs z validation test =====')
    vt.comvDist_vs_z([lcDir_extrap], steps=allSteps[::-1], plotMode=plotMode,
                     outDir = figDir)
        
        # ----------------------------------------------------------------------
       
    if(pmode == 'halos'):
        if(0):
            print('\n\n ==== pairwise separation validation test =====')
            #vt.find_pairwise_separation(lcDir_interp, 'interp', pairwiseDataDir, 
            #                            steps=allSteps,
            #                            solverMode = 'backward', fofDir=fofDir)
            vt.find_pairwise_separation(lcDir_extrap, 'extrap', pairwiseDataDir, 
                                        steps=allSteps,
                                        solverMode = 'forward', fofDir=fofDir)
        print('\n\n ==== plotting pairwise separation validation test =====')
        plotMode='show'
        vt.plot_pairwise_separation(pairwiseDataDir, 'interp', steps=allSteps, 
                                    plotMode=plotMode, outDir=figDir)
        vt.plot_pairwise_separation(pairwiseDataDir, 'extrap', steps=allSteps, 
                                    plotMode=plotMode, outDir=figDir)
    

# ==================================================================================================
# ==================================================================================================


def allTests(lcDir_interp, lcDir_extrap, snapshotDir, figDir, trajectoryDataDir, trajectoryPlotDir,
             duplicateDataDir, pmode, snapSubdirs, plotMode, smode, cmap):
   
    if(0):
        matchingTests(lcDir_interp, lcDir_extrap, snapshotDir, figDir, trajectoryDataDir, trajectoryPlotDir,
                      duplicateDataDir, pairwiseDataDir, pmode, snapSubdirs, plotMode, smode, cmap)
    
    comparisonTests(lcDir_interp, lcDir_extrap, snapshotDir, '', figDir,
                    duplicateDataDir, '', pmode, snapSubdirs, plotMode, smode, cmap)

    
# ==================================================================================================
# ==================================================================================================

   
def particles():

    #allTests(lcDir_interp='/projects/DarkUniverse_esp/jphollowed/outerRim/lightcone_downsampled_octant_new', 
    #         lcDir_extrap='/projects/DarkUniverse_esp/jphollowed/outerRim/lightcone_downsampled_octant', 
    allTests(lcDir_interp='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_full_octant', 
             lcDir_extrap='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_downsampled_octant_extrap', 
             snapshotDir='/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Particles', 
             figDir='/home/hollowed/lc_validation/', 
             trajectoryDataDir='/home/hollowed/lc_validation/trajData', 
             trajectoryPlotDir='/home/hollowed/lc_validation/trajPlots', 
             duplicateDataDir='/home/hollowed/lc_validation/duplData' ,
             pmode='particles', 
             snapSubdirs=True, 
             plotMode='show',
             smode='forward',
             cmap=None)


def particles_reversed():

    allTests(lcDir_interp='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_downsampled_octant_backward', 
             lcDir_extrap='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_downsampled_octant_extrap_backward', 
             snapshotDir='/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Particles', 
             figDir='/home/hollowed/lc_reversed_validation/', 
             trajectoryDataDir='/home/hollowed/lc_reversed_validation/trajData', 
             trajectoryPlotDir='/home/hollowed/lc_reversed_validation/trajPlots', 
             duplicateDataDir='/home/hollowed/lc_reversed_validation/duplData' ,
             pmode='particles', 
             snapSubdirs=True, 
             plotMode='save', 
             smode='backward',
             cmap=None)


def halos():
   
    if(0):
        matchingTests(lcDir_interp='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_halos', 
                      lcDir_extrap='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_halos_extrap', 
                      snapshotDir='/projects/SkySurvey/rangel/lc_test_data/temp', 
                      figDir='/home/hollowed/lc_halos_validation/', 
                      trajectoryDataDir='/home/hollowed/lc_halos_validation/trajData', 
                      trajectoryPlotDir='/home/hollowed/lc_halos_validation/trajPlots', 
                      duplicateDataDir='/home/hollowed/lc_halos_validation/duplData' ,
                      pmode='halos',
                      snapSubdirs=False,
                      plotMode='save', 
                      smode='backward',
                      cmap=None)

    comparisonTests(lcDir_interp='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_halos', 
                    lcDir_extrap='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_halos_extrap_oldSolver', 
                    snapshotDir='/projects/SkySurvey/rangel/lc_test_data/temp', 
                    fofDir = '/projects/SkySurvey/rangel/lc_test_data/b0168',
                    figDir='/home/hollowed/lc_halos_validation/', 
                    duplicateDataDir='/home/hollowed/lc_halos_validation/duplData',
                    pairwiseDataDir = '/home/hollowed/lc_halos_validation/pairData',
                    pmode='halos',
                    snapSubdirs=False,
                    plotMode='save', 
                    smode='backward',
                    cmap=None)
