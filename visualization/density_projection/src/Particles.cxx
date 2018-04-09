// Particles.cxx 
//
// Class whcih reads input cosmo file and returns particle attributes
//

#include "Particles.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <assert.h>

#include "GenericIO.h"
using namespace gio;

Particles::Particles(std::string inputFile, int command, int species, int fileType)
{

  // Set commmand (0 for density, 1 for uu)
  assert(command == 0 || command == 1);
  vtype = command;

  // Set particle type (0 for dm, 1 for baryons)
  assert(species == 0 || species == 1);
  ptype = species;

  // Read particle data (fileType of 0 for GIO, 1 for Cosmo)
  filename = inputFile;
  int success; 
  if( fileType == 0){ success = ReadGIOFile(); }
  if( fileType == 1){ success = ReadCosmoFile(); }

  assert(success == 0);

  // Print some stats
  PrintStats();

}

int Particles::ReadGIOFile()
{
  
  std::cout << "Opening file: " << filename << std::endl;
  GenericIO reader(MPI_COMM_SELF, filename);
  reader.openAndReadHeader(GenericIO::MismatchRedistribute);

  // Determine particle count and adjust array size (this will include total
  // particle count, but we will only read dm or baryon particles, dictated 
  // by the value of ptype)
  size_t nTotal = 0;
  int nRanks = reader.readNRanks();
  size_t current_size;
  for (int j=0; j<nRanks; ++j) {
    current_size = reader.readNumElems(j);
    nTotal = current_size > nTotal ? current_size : nTotal;
  }
  nTotal +=10;
  std::cout<< "max size: " << nTotal << std::endl;

  // start reading 
  std::vector<float> xx0;
  std::vector<float> yy0;
  std::vector<float> zz0;
  std::vector<int64_t> id0;
  
  xx0.resize(nTotal);
  yy0.resize(nTotal);
  zz0.resize(nTotal);
  id0.resize(nTotal);
  
  reader.addVariable("x", &xx0[0]);
  reader.addVariable("y", &yy0[0]);
  reader.addVariable("z", &zz0[0]);
  reader.addVariable("vx", &xx0[0]);
  reader.addVariable("vy", &yy0[0]);
  reader.addVariable("vz", &zz0[0]); 
  reader.addVariable("id", &id0[0]);
  
  for (int j=0; j<nRanks; ++j) {
    
    size_t current_size = reader.readNumElems(j);
    reader.readData(j, false);
   
    for(int k=0; k<current_size; ++k){ 
      xx.push_back(xx0[k]);
      yy.push_back(yy0[k]);
      zz.push_back(zz0[k]);
      k++;
    }
    if(j == 0){ std::cout << " Done reading on rank 0" << std::endl; }
  }

  // Close file 
  reader.close();
  return 0;
}


int Particles::ReadCosmoFile()
{

#ifdef TESTING

  int nPdim = 128;

  int nTotal = nPdim*nPdim*nPdim; 
  xx.resize(nTotal);
  yy.resize(nTotal);
  zz.resize(nTotal);
  hh.resize(nTotal);
  vv.resize(nTotal);
 
  float dxp = 50.0/nPdim;
  float dyp = 50.0/nPdim;
  float dzp = 66.6/nPdim;
 
  nParticle = 0;
  for (int i=0; i<nPdim; ++i) {
    for (int j=0; j<nPdim; ++j) {
      for (int k=0; k<nPdim; ++k) {
        xx[nParticle] = i*dxp; 
        yy[nParticle] = j*dyp;
        zz[nParticle] = k*dzp;
        hh[nParticle] = 1.0f;
        vv[nParticle] = 1.0f;
        nParticle++;
      }
    }
  }

  std::cout << " Done setting up uniform ditribution with " << nParticle << " particles " << std::endl; 

#else

  FILE *inputFile = fopen(filename.c_str(), "rb");
  if (!inputFile) {
    std::cout << " ERROR: Could not open inputFile " << filename << std::endl;
    return -1;
  }

  // Determine particle count and adjust array size (this will include total
  // particle count, but we will only read dm or baryon particles, dictated 
  // by the value of ptype)
  fseek(inputFile, 0, SEEK_END);
  int nTotal = ftell(inputFile)/(12*sizeof(POSVEL_T)+sizeof(ID_T));
  xx.resize(nTotal);
  yy.resize(nTotal);
  zz.resize(nTotal);
  hh.resize(nTotal);
  vv.resize(nTotal);

  // Do the reading now 
  POSVEL_T xx0, yy0, zz0, vx0, vy0, vz0, mm0, uu0, hh0, mu0, rho0, phi0, vv0;
  POSVEL_T psi0;
  ID_T id0;
  rewind(inputFile);
  nParticle = 0;
  std::cout << " Reading from file " << filename << " ... " << std::endl;
  for (int i=0; i<nTotal; ++i) {
    fread(&xx0, sizeof(POSVEL_T), 1, inputFile);
    fread(&vx0, sizeof(POSVEL_T), 1, inputFile);
    fread(&yy0, sizeof(POSVEL_T), 1, inputFile);
    fread(&vy0, sizeof(POSVEL_T), 1, inputFile);
    fread(&zz0, sizeof(POSVEL_T), 1, inputFile);
    fread(&vz0, sizeof(POSVEL_T), 1, inputFile);
    fread(&mm0, sizeof(POSVEL_T), 1, inputFile);
    fread(&uu0, sizeof(POSVEL_T), 1, inputFile);
    fread(&hh0, sizeof(POSVEL_T), 1, inputFile);
    fread(&mu0, sizeof(POSVEL_T), 1, inputFile);
    fread(&rho0, sizeof(POSVEL_T), 1, inputFile);
    fread(&phi0, sizeof(POSVEL_T), 1, inputFile);
    fread(&id0, sizeof(ID_T), 1, inputFile);
    if (id0%2 != 1 && ptype == 0) { // This is a dm particle 
      if (vtype == 0) psi0 = rho0;
      else if (vtype == 1) psi0 = uu0;
      xx[nParticle] = xx0;
      yy[nParticle] = yy0;
      zz[nParticle] = zz0;
      hh[nParticle] = hh0;
      vv[nParticle] = mm0*psi0/rho0;
      nParticle++;
    }
    if (id0%2 == 1 && ptype == 1) { // This is a baryon 
      if (vtype == 0) psi0 = rho0;
      else if (vtype == 1) psi0 = uu0;
      xx[nParticle] = xx0;
      yy[nParticle] = yy0;
      zz[nParticle] = zz0;
      hh[nParticle] = hh0;
      vv[nParticle] = mm0*psi0/rho0;
      nParticle++;
    }
  }
  std::cout << " Done reading " << nParticle << " particles from a total set of " << 
               nTotal << " dm+baryon particles " << std::endl;

  // Close file 
  fclose(inputFile);

#endif

  return 0;
}

void Particles::PrintStats()
{

  // Collect min and maximum values
  POSVEL_T xxmin = 1.e8; POSVEL_T xxmax = -1.e8;
  POSVEL_T yymin = 1.e8; POSVEL_T yymax = -1.e8;
  POSVEL_T zzmin = 1.e8; POSVEL_T zzmax = -1.e8;
  POSVEL_T hhmin = 1.e8; POSVEL_T hhmax = -1.e8;
  POSVEL_T vvmin = 1.e8; POSVEL_T vvmax = -1.e8;
  for (int i=0; i<nParticle; ++i) {
    xxmin = std::min(xxmin, xx[i]); xxmax = std::max(xxmax, xx[i]);
    yymin = std::min(yymin, yy[i]); yymax = std::max(yymax, yy[i]);
    zzmin = std::min(zzmin, zz[i]); zzmax = std::max(zzmax, zz[i]);
    hhmin = std::min(hhmin, hh[i]); hhmax = std::max(hhmax, hh[i]);
    vvmin = std::min(vvmin, vv[i]); vvmax = std::max(vvmax, vv[i]);
  }
 
  // Print information to screen
  std::cout << " xmin: " << xxmin << " xmax: " << xxmax << std::endl;
  std::cout << " ymin: " << yymin << " ymax: " << yymax << std::endl;
  std::cout << " zmin: " << zzmin << " zmax: " << zzmax << std::endl; 
  std::cout << " hmin: " << hhmin << " hmax: " << hhmax << std::endl;
  std::cout << " vmin: " << vvmin << " vmax: " << vvmax << std::endl;
  std::cout << std::endl;

}

Particles::~Particles()
{

}

