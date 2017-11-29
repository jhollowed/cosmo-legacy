#!/bin/sh

# Cobalt parameters for job scheduling
# More detail at https://www.alcf.anl.gov/user-guides/using-cobalt-cooley
NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=$((NODES*1))
TS=$1

# Light Cone parameters
INPUT=/projects/SkySurvey/heitmann/AlphaQ/lightcones/lc_galaxies
OUTPUT=/projects/SkySurvey/dkorytov/protoDC2/gal_lc_final
STEPS="382 373 365 355 347 338 331 323 315 307 300 293 286 279 272 266 259"

# Run expression
mpirun -f $COBALT_NODEFILE -n $PROCS ./halo_lc_cutout $INPUT  $OUTPUT $STEPS
