// 
// ProjectDensity.cxx 
//
// Program to perform a 3D projection of SPH density from a single HACC domain.
//

#include "Particles.h"
#include "ChainingMesh.h"
#include "CloudsInCells.h"
#include "Skewer.h"
#include <omp.h>
#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <sstream>

#define MIN_VAL 1.0e-30f

int main(int argc, char *argv[])
{

  // Read user arguments: 
  //
  // cosmofile: input particle snapshot file in .cosmo format
  // outfile: the output destination
  // x,y,zmin and x,y,zmax: the comoving cartesian domain of the final render in Mpc/h
  // mesh_width: symmetric width of mesh cells in comoving Mpc/h
  // x,y,zobs: comoving cartesian position of the observer in Mpc/h
  // depth: comoving depth distance, in Mpc/h, to which the observer can "see" (particles 
  //        beyond this depth will not contribute to pizel densities)
  // xfov: the observers field of view in the x-direction (y- and z-direction fov assumed 
  //       symmetric) in degrees
  // fobs:
  // theta:
  // phi:
  // xpix: pixels in the horizontal direction in the plane of observation
  // ypix: pixels in the vertical direction in the plane of observation
  // zsamp: sampling frequency of skewer density inerpolation points
  
  MPI_Init(&argc, &argv);

  if(argc < 23) {
    std::cerr << "USAGE: " << argv[0] << " <inputFile> <outfile> <xmin> <xmax> <ymin>" <<
                                         " <ymax> <zmin> <zmax> <mesh_width> <xobs> <yobs>" << 
                                         " <zobs> <depth> <xfov> <fobs> <theta> <phi> " << 
                                         " <xpix> <ypix> <zsamp> <ptype> <fileType>" << std::endl; 
    return -1;
  }

  char inputFile[200], outFile[200];
  sprintf(inputFile, argv[1]);
  sprintf(outFile,   argv[2]);
  float xmin  = atof(argv[3]);
  float xmax  = atof(argv[4]);
  float ymin  = atof(argv[5]);
  float ymax  = atof(argv[6]);
  float zmin  = atof(argv[7]);
  float zmax  = atof(argv[8]);
  float dmesh = atof(argv[9]);
  float xobs  = atof(argv[10]);
  float yobs  = atof(argv[11]);
  float zobs  = atof(argv[12]);
  float depth = atof(argv[13]);
  float xfov  = atof(argv[14]);
  float fobs  = atof(argv[15]);
  float theta = atof(argv[16]);
  float phi   = atof(argv[17]);
  int nxpix   = atoi(argv[18]);
  int nypix   = atoi(argv[19]);
  int nsamp   = atoi(argv[20]);
  int ptype   = atoi(argv[21]);
  int ftype   = atoi(argv[22]);
  
  // Total number of pixels
  int nxypix = nxpix*nypix;

  // Box centre (used to pivot rotations)
  float xr = 0.5f*(xmin + xmax);
  float yr = 0.5f*(ymin + ymax);
  float zr = 0.5f*(zmin + zmax);

  // Read particle positions, smoothing lengths, and densities (will only keep track of 
  // particles noted as ptype)
  Particles *p = new Particles(std::string(inputFile), 0, ptype, ftype);

  int nParticle = p->NumParticles();
  float *xx   = p->ExtractX();
  float *yy   = p->ExtractY();
  float *zz   = p->ExtractZ();
  float *hh   = p->ExtractH();
  float *vv   = p->ExtractV();

  // Construct cloud in cells mesh that keeps track of dm particles in each cell
  CloudsInCells *cic = new CloudsInCells(ptype, xmin, xmax, ymin, ymax, zmin, zmax, dmesh,
                                         nParticle, xx, yy, zz, hh, vv);
  // Construct chaining mesh that keeps track of baryons in each cell 
  // (particle plus smoothing sphere) 
  ChainingMesh *cm = new ChainingMesh(ptype, xmin, xmax, ymin, ymax, zmin, zmax, dmesh, 
                                      nParticle, xx, yy, zz, hh, vv);

  // Sample density along each skewer that starts at (xobs, yobs, zobs) and ends on the pixel plane.

  int nthreads = 1;
#ifdef _OPENMP
//  nthreads = 32;
  nthreads = omp_get_max_threads();
#endif

  Skewer *skewer = new Skewer(xobs, yobs, zobs, xr, yr, zr, depth, xfov, fobs, theta, 
                              phi, nxpix, nypix, nsamp, nthreads);

  std::vector<float> pixels;
  pixels.clear();
  pixels.resize(nxypix);

#ifdef _OPENMP
  #pragma omp parallel for num_threads(nthreads) 
#endif
  for (int ipix=0; ipix<nxpix; ipix++) {
#ifdef _OPENMP
    int nt = omp_get_thread_num(); 
#else
    int nt = 0;
#endif
    for (int jpix=0; jpix<nypix; jpix++) {

      // Determine interpolation points for this skewer
      skewer->InterpolationPoints(nt, ipix, jpix);
      float *xxs = skewer->ExtractX(nt);
      float *yys = skewer->ExtractY(nt);
      float *zzs = skewer->ExtractZ(nt);

      // Feed skewer into the chaining mesh and compute column density
      int ijpix = ipix*nypix + jpix;
      if(ptype == 0){
          pixels[ijpix] = cic->ColumnDensity(xxs, yys, zzs, nsamp);
      } else if(ptype == 1){
          pixels[ijpix] = cm->ColumnDensity(xxs, yys, zzs, nsamp);
      }
    }
  }

  // Take the logarithm of the result
  for (int i=0; i<nxypix; ++i) {
    pixels[i] = log10(std::max(pixels[i], MIN_VAL));
  }

  // Save the result
  FILE *fout = fopen(outFile, "wb");
  for (int i=0; i<nxypix; ++i) fwrite(&pixels[i], sizeof(float), 1, fout);
  fclose(fout);

  // Cleanup
  delete skewer;
  delete cic;
  delete cm;
  delete p;
  
  MPI_Finalize();
  return 0;
}

