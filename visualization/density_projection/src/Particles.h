// Particles.h 
//
// Class whcih reads input cosmo file and returns particle attributes 
//

#ifndef Particles_h
#define Particles_h

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

class Particles
  {

    public:

      Particles(std::string inputFile, int command1, int command2, int command3);
      
      int ReadCosmoFile();
      void PrintStats();

      ~Particles();

      int NumParticles() { return nParticle; }
      float* ExtractX()  { return &xx[0]; }
      float* ExtractY()  { return &yy[0]; }
      float* ExtractZ()  { return &zz[0]; }
      float* ExtractH()  { return &hh[0]; }
      float* ExtractV()  { return &vv[0]; }

    protected: 

      typedef float POSVEL_T;
      typedef int64_t ID_T;

      // Particle file
      std::string filename;
      
      // Determines what particle types to read (0 for dm, 1 for bayons)
      int ptype;

      // Determines what attribute to return (0 for rho, 1 for uu) (relevant for baryons)
      int projMode;

      // Indicates what simulation run type this input data came from (0 for gravity only, 1 for hydro)
      int runType;
      

      // Particle arrays 
      int nParticle;
      std::vector<float> xx;
      std::vector<float> yy;
      std::vector<float> zz;
      std::vector<float> hh;
      std::vector<float> vv; // This will be the attribute we are projecting (decided by projMode)

  } ;
#endif

