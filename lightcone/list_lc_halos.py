# Joe Hollowed
# CPAC 2018

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))

import pdb
import glob
import numpy as np
import genericio as gio
import lc_interpolation_validation as iv

def list_halos(lcDir, soDir, outDir, maxStep, minStep, massCut=1e14, outFrac=0.01, numFiles=1):

    '''
    This function generates a list of halo identifiers, and positions, in a text 
    file, in the format expected by Use Case 2 of the lightcone cutout code
    at https://github.com/jhollowed/cosmo-cutout. That is, it reads the output of a
    halo lightcone run, finds all halos above some mass cut, and writes them to
    a text file with halos printed per-row, as such:
    
    output.txt:
    HALO_ID_1 x1 y1 z1
    HALO_ID_2 x2 y2 z2
    HALO_ID_3 x3 y3 z3
    ...
    
    where HALO_ID_N can be anything (it will ultimately be read into a string type variable), 
    but typically contains the fof halo tag, plus some other meta data. In this function
    The HALO_ID_N's are written as
    
    {fof_halo_tag}_z{halo_redshift}_MSOD{sod_halo_mass}
    
    Params:
    :param lcDir: top-level directory of a halo lightcone, where the subdirectory 
                  structure is expected to match that described in section 4.5 (fig 7)
                  of the Creating Lightcones in HACC document (step-wise subdirectories 
                  expected). The subdrectory prefix is assumed to be 'lcHalos'
                  Furthermore, It is expected that this lightcone was built using the
                  interpolation lightcone driver, and thus the 'id' field is expected
                  to contain merger tree indices.
    :param soDir: top-level directory of a halo SOD catalog, matching the halos found in
                  the merger trees at mtDir. Step-wise subdirectoires are expected with 
                  the form 'STEPXXX'
    :para outDir: the output directory for the resultant text file
    :param maxStep: The largest (lowest redshift) lightcone shell to read in
    :param minStep: The smallest (highest redshift) lightcone shell to read in
    :param massCut: The minimum halo mass to write out to the text files
    :param outFrac: The fraction of the identified halos to actually output
    :param numFiles: How many text files to write. That is, if 30 halos are found in
                     the lightcone at lcDir, between minStep and maxStep, and numFiles=3,
                     then three text files will be written out, each containing 10 of the 
                     found halos. This option is intended to allow for separate cutout
                     runs being submitted in parallel, each handling a subset of all the
                     desired halos
    '''

    # get lightcone shells (step/snapshot numbers) matching those desired 
    # by minStep and maxStep
    lcHaloSubdirs = glob.glob('{}/*'.format(lcDir))
    steps = np.array(sorted([int(s.split('Halos')[-1]) for s in lcHaloSubdirs]))
    steps = steps[np.logical_and(steps >= minStep, steps <= maxStep)]
  
    # arrays to hold halos in the lc found above the desired massCut (to be written out)
    write_ids = np.array([])
    write_x = np.array([])
    write_y = np.array([])
    write_z = np.array([])

    # loop over lightcone shells
    for i in range(len(steps)-1):
        
        step = steps[i]
        if(step == 499): continue

        print('\n---------- working on step {} ----------'.format(step))
       
        print('reading lightcone')
        lc_file = '{1}/lcHalos{0}/lc_intrp_halos.{0}'.format(step, lcDir)
        lc_tags = np.squeeze(gio.gio_read(lc_file, 'id'))
        lc_x = np.squeeze(gio.gio_read(lc_file, 'x'))
        lc_y = np.squeeze(gio.gio_read(lc_file, 'y'))
        lc_z = np.squeeze(gio.gio_read(lc_file, 'z'))
        lc_a = np.squeeze(gio.gio_read(lc_file, 'a'))
        # get halo redshifts and mask halo fof tags
        lc_redshift = 1/lc_a - 1
        lc_tags = (lc_tags * np.sign(lc_tags)) & 0x0000ffffffffffff 

        print('reading SO catalog')
        #sod_file = '{1}/STEP{0}/m000-{0}.sodproperties'.format(step, soDir)
        #sod_file = '{1}/b0168/STEP{0}/m000-{0}.fofproperties'.format(step, soDir.split('/M200')[0])

        # matching issues... lets just use these fo masses for now

        sod_file = '/projects/SkySurvey/rangel/lc_test_data/b0168/STEP{0}/m000-{0}.fofproperties'.format(steps[i+1])
        sod_tags = np.squeeze(gio.gio_read(sod_file, 'fof_halo_tag'))
        #sod_mass = np.squeeze(gio.gio_read(sod_file, 'sod_halo_mass'))
        sod_mass = np.squeeze(gio.gio_read(sod_file, 'fof_halo_mass'))

        # Now we match to get the halo masses, with the matching done in the
        # folloing order:
        # lc masked fof_halo_tag > sod fof_halo_tag
        # sod fof_halo_tag > sod_halo_mass

        print('matching lightcone to so catalog to retrieve mass')
        lc_to_sod = iv.search_sorted(sod_tags, lc_tags)

        # make sure that worked
        if(np.sum(lc_to_sod == -1) != 0):
            raise Exception('{0}% of lightcone halos not found in SO catalog. '\
                            'Maybe passed wrong files?'
                            .format(np.sum(lc_to_sod==-1)/float(len(lc_to_sod)) * 100))  

        lc_mass = sod_mass[lc_to_sod]

        # do mass cutting
        mass_mask = lc_mass >= massCut
        lc_tags = lc_tags[mass_mask]
        lc_redshift = lc_redshift[mass_mask]
        lc_mass = lc_mass[mass_mask]
        lc_x = lc_x[mass_mask]
        lc_y = lc_y[mass_mask]
        lc_z = lc_z[mass_mask]

        # make halo identifier strings (see docstrings at function header)
        lc_ids = ['{0}_z{1}_MFOF{2}'.format(lc_tags[i], lc_redshift[i], lc_mass[i]) 
                  for i in range(len(lc_tags))]

        # add these halos to write-out arrays
        print('Found {0} halos ({1:.5f}% of all) above mass cut of {2}'
              .format(np.sum(mass_mask), (np.sum(mass_mask)/float(len(mass_mask)))*100, massCut))
        print('Appending halo data to write-out arrays')
        write_ids = np.hstack([write_ids, lc_ids]) 
        write_x = np.hstack([write_x, lc_x]) 
        write_y = np.hstack([write_y, lc_y]) 
        write_z = np.hstack([write_z, lc_z]) 
    
    # Do downsampling according to outFrac arg
    print('\nDownsampling {0}% of {1} total halos'.format(outFrac*100, len(write_ids)))
    dsampling = np.random.choice(np.arange(len(write_ids)), int(len(write_ids)*outFrac), replace=False)
    write_ids = write_ids[dsampling]
    write_x = write_x[dsampling]
    write_y = write_y[dsampling]
    write_z = write_z[dsampling]

    # Now do writing to text file(s)
    print('\nDone, obtained {0} total halos to write across {1} text files'
          .format(len(write_ids), numFiles))
    write_masks = np.array_split(np.arange(len(write_ids)), numFiles)
    for j in range(numFiles):
        wm = write_masks[j]
        next_file = open("{0}/lc_halos_{1}.txt".format(outDir, j), 'w')
        print('writing {0} halos to file {1}'.format(len(wm), j+1))
        for n in range(len(wm)):
            next_file.write('{0} {1} {2} {3}\n'.format(write_ids[wm][n], 
                                                     write_x[wm][n], 
                                                     write_y[wm][n], 
                                                     write_z[wm][n]))
        next_file.close()
    print('Done')
        

# =================================================================================================


def list_alphaQ_halos(maxStep=487, minStep=247, massCut=1e14, outFrac=0.01, numFiles=1):
    
    '''
    This function runs list_halos with data paths predefined for AlphaQ.
    Function parameters are as given in the docstrings above for list_halos
    '''

    list_halos(lcDir='/projects/DarkUniverse_esp/jphollowed/alphaQ/lightcone_halos',
               soDir='/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Halos/M200',
               outDir='/home/hollowed/cutout_run_dirs/alphaQ/cutout_alphaQ_full',
               maxStep=maxStep, minStep=minStep, massCut=massCut, outFrac=outFrac, numFiles=numFiles)
