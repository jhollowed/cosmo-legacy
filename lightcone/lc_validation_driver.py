'''
This file runs all of the validation test written in lc_interpolation_validation with ease
'''

import lc_interpolation_validation as vt

def allTests(lcDir_interp, lcDir_extrap, snapshotDir, histFigDir, trajectoryDir, trajectoryPlotDir, 
             duplicateDir, duplicatePlotDir, replicationPlotDir, comvDist_vs_redshift_PlotDir, 
             pmode, snapSubdirs, plotMode, cmap):
    
    print('\n\n ==== Histograms validation test =====')
    vt.lightconeHistograms(lcDir_interp, lcDir_extrap, step=442, rL=256, 
                           plotMode=plotMode, outDir=histFigDir)

    print('\n\n ==== Particle trajectory validation test =====')
    diffRanges = ['min', 'med', 'max']
    for diff in diffRanges:
        vt.saveLightconePathData(lcDir_extrap, lcDir_interp, snapshotDir, trajectoryDir,
                              rL=256, diffRange=diff, snapshotSubdirs=snapSubdirs, mode=pmode)

    print('\n\n ==== Plotting particle trajectory validation test =====')
    diffRanges = ['min', 'med', 'max']
    for diff in diffRanges:
        vt.plotLightconePaths(trajectoryDir, diffRange=diff, plotMode=plotMode, outDir=trajectoryPlotDir)
    print('\n\n ==== Finding duplicates validation test =====')
    if(pmode == 'halos'):
        vt.findDuplicates(lcDir_interp, steps=[432, 442, 453], lcSuffix='interp', outDir=duplicateDir,
                          mode=pmode, mergerTreeDir=snapshotDir)
        vt.findDuplicates(lcDir_extrap, steps=[432, 442, 453], lcSuffix='extrap', outDir=duplicateDir,
                          mode=pmode, mergerTreeDir=snapshotDir)
    else:
        vt.findDuplicates(lcDir_interp, steps=[432, 442], lcSuffix='interp', outDir=duplicateDir,
                          mode=pmode)
        vt.findDuplicates(lcDir_extrap, steps=[432, 442], lcSuffix='extrap', outDir=duplicateDir,
                          mode=pmode)

    print('\n\n ==== Plotting duplicates validation test =====')
    vt.compareDuplicates(duplicateDir, steps=[432, 442], lcSuffix=['interp', 'extrap'],
                         plotMode=plotMode, outDir=duplicatePlotDir)
    
    print('\n\n ==== Compare Replications validation test =====')
    vt.compareReps(lcDir_interp, lcDir_extrap, step=442, plotMode=plotMode, outDir=replicationPlotDir)

    print('\n\n ==== Comv Dist vs z validation test =====')
    vt.comvDist_vs_z([lcDir_interp, lcDir_extrap], steps=[487, 475, 464, 453, 442], plotMode=plotMode,
                     outDir = comvDist_vs_redshift_PlotDir)

   
def particles():

    allTests(lcDir_interp='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_downsampled_octant_new', 
             lcDir_extrap='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_downsampled_octant_extrap', 
             snapshotDir='/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Particles', 
             histFigDir='/home/hollowed/lc_validation/hists', 
             trajectoryDir='/home/hollowed/lc_validation/trajData', 
             trajectoryPlotDir='/home/hollowed/lc_validation/trajPlots', 
             duplicateDir='/home/hollowed/lc_validation/duplData' ,
             duplicatePlotDir='/home/hollowed/lc_validation/duplPlots', 
             replicationPlotDir='/home/hollowed/lc_validation/replPlots', 
             comvDist_vs_redshift_PlotDir='/home/hollowed/lc_validation/comvDist_z_plots', 
             pmode='particles', 
             snapSubdirs=True, 
             plotMode='save', 
             cmap=None)

def halos():

    allTests(lcDir_interp='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_halos', 
             lcDir_extrap='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_halos_extrap', 
             snapshotDir='/projects/SkySurvey/rangel/lc_test_data/temp', 
             histFigDir='/home/hollowed/lc_halos_validation/hists', 
             trajectoryDir='/home/hollowed/lc_halos_validation/trajData', 
             trajectoryPlotDir='/home/hollowed/lc_halos_validation/trajPlots', 
             duplicateDir='/home/hollowed/lc_halos_validation/duplData' ,
             duplicatePlotDir='/home/hollowed/lc_halos_validation/duplPlots', 
             replicationPlotDir='/home/hollowed/lc_halos_validation/replPlots', 
             comvDist_vs_redshift_PlotDir='/home/hollowed/lc_halos_validation/comvDist_z_plots', 
             pmode='halos',
             snapSubdirs=False,
             plotMode='save', 
             cmap=None)
