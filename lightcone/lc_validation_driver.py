'''
This file runs all of the validation test written in lc_interpolation_validation with ease
'''
import lc_interpolation_validation as vt

def allTests(lcDir_interp, lcDir_extrap, snapshotDir, histFigDir, trajectoryDir, trajectoryPlotDir, 
             duplicateDir, duplicatePlotDir, replicationPlotDir, comvDist_vs_redshift_PlotDir, 
             plotMode, cmap):
    if(0): 
        print('\n\n ==== Histograms validation test =====')
        vt.lightconeHistograms(lcDir_interp, lcDir_extrap, step=442, rL=256, 
                               plotMode=plotMode, outDir=histFigDir)

        print('\n\n ==== Particle trajectory validation test =====')
        diffRanges = ['min', 'med', 'max']
        for diff in diffRanges:
            vt.saveLightconePathData(lcDir_extrap, lcDir_interp, snapshotDir, trajectoryDir,
                                  rL=256, diffRange=diff, snapshotSubdirs=True)

        print('\n\n ==== Plotting particle trajectory validation test =====')
        diffRanges = ['min', 'med', 'max']
        for diff in diffRanges:
            vt.plotLightconePaths(trajectoryDir, diffRange=diff, plotMode=plotMode, outDir=trajectoryPlotDir)

        print('\n\n ==== Finding duplicates validation test =====')
        vt.findDuplicates(lcDir_interp, steps=[432, 442], lcSuffix='interp', outDir=duplicateDir)
        vt.findDuplicates(lcDir_extrap, steps=[432, 442], lcSuffix='extrap', outDir=duplicateDir)
    
    print('\n\n ==== Plotting duplicates validation test =====')
    vt.compareDuplicates(duplicateDir, steps=[432, 442], lcSuffix=['interp', 'extrap'],
                         plotMode=plotMode, outDir=duplicatePlotDir)
    
    print('\n\n ==== Compare Replications validation test =====')
    vt.compareReps(lcDir_interp, lcDir_extrap, step=442, plotMode=plotMode, outDir=replicationPlotDir)

    print('\n\n ==== Comv Dist vs z validation test =====')
    vt.comvDist_vs_z([lcDir_interp, lcDir_extrap], steps=[487, 475, 464, 453, 442], plotMode=plotMode,
                     outDir = comvDist_vs_redshift_PlotDir)

   



