// ChainingMesh.h 
//
// Class which splits the domain into chaining mesh cells where each cell
// contains a list of all particles whose smoothing radius intersects the cell.
//

#ifndef ChainingMesh_h
#define ChainingMesh_h

#include <vector>

class ChainingMesh
  {

    public:

      ChainingMesh(int ptype,
                   float xmin, float xmax,       // Chaining mesh extents
                   float ymin, float ymax,
                   float zmin, float zmax,
                   float dlen,                   // Cell width on a side
                   int np,                       // Number of particles
                   float *xxloc, float *yyloc,   // Particle attributes
                   float *zzloc, float *hhloc,
                   float *vvloc);

      ~ChainingMesh();

      void setupInteractionLists();
      int CellIndex(int ii, int jj, int kk);
      bool CheckIntersection(float xxp, float yyp, float zzp, float hhp, int ii, int jj, int kk);
      float ColumnDensity(float *xxs, float *yys, float *zzs, int nsamp);

    protected: 

      double SPHInterpolation(float xxi, float yyi, float zzi, float xx0, float yy0, float zz0, float hh0, float vv0);
      double SPHKernel(double eta, double h);

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

      // Each chaining mesh cell contains an array of particle indices
      std::vector<int> *ipi;

  } ;
#endif

