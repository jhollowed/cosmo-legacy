// CloudsInCells.cxx 
//
// Class which splits the domain into cells and distributes particle masses among
// those cell vertices
//

#ifndef CloudsinCells_h
#define CloudsinCells_h

#include <vector>

class CloudsInCells
  {

    public:

      CloudsInCells(int ptype,
                    float xmin, float xmax,
                    float ymin, float ymax,
                    float zmin, float zmax,
                    float dlen,
                    int np,
                    float *xxloc, float *yyloc,
                    float *zzloc, float *hhloc,
                    float *vvloc);

      ~CloudsInCells();

      void DistributeToCells();
      int CellIndex(int ii, int jj, int kk);
      float ColumnDensity(float *xxs, float *yys, float *zzs, int nsamp);

    protected: 
      
      // Particle and chaining mesh cell counts
      int nParticle;
      int nMeshX;
      int nMeshY;
      int nMeshZ;
      int nMesh;

      // Chaining mesh boundaries
      float x0, x1, y0, y1, z0, z1;
      float Lx, Ly, Lz;
      float dr; 

      // Particle attributes
      float *xx;
      float *yy;
      float *zz;
      float *hh;
      float *vv;

      // Each CIC cell contains an array of particle indices and particle
      // density contributions 
      std::vector<int> *cellParticles;
      std::vector<float> *cellDensity;

  } ;
#endif
