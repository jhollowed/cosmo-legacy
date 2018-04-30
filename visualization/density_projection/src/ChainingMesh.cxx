// ChainingMesh.cxx 
//
// Class which splits the domain into chaining mesh cells where each cell
// contains a list of all particles whose smoothing radius intersects the cell.
//

#include "ChainingMesh.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h> 
#include <assert.h>

#define ETAMAX 1.0

ChainingMesh::ChainingMesh(int ptype, float xmin, float xmax, float ymin, float ymax, float zmin, 
                           float zmax, float dlen, int np, float *xxloc, float *yyloc, float *zzloc, 
                           float *hhloc, float *vvloc)
{
  
  if(ptype == 0){ return; } 
    
  // Set chaining mesh boundaries and determine how many cells per dimension there are
  x0 = xmin; x1 = xmax; Lx = xmax - xmin;
  y0 = ymin; y1 = ymax; Ly = ymax - ymin;
  z0 = zmin; z1 = zmax; Lz = zmax - zmin;
  dr = dlen;
  nMeshX = static_cast<int>(ceil(Lx/dr));
  nMeshY = static_cast<int>(ceil(Ly/dr));
  nMeshZ = static_cast<int>(ceil(Lz/dr));
  nMesh  = nMeshX*nMeshY*nMeshZ;

  // Set data arrays
  nParticle = np;
  xx = xxloc;
  yy = yyloc;
  zz = zzloc;

  // Instantiate array holding particle indices for each cell
  ipi = new std::vector<int>[nMesh];

  // Setup interaction lists within ipi so that each array contains the
  // list of all particles indices whose smoothing sphere intersects with the cell.
  setupInteractionLists();

}

void ChainingMesh::setupInteractionLists()
{

  std::cout << " Setting up chaining mesh with nx: " << nMeshX << " ny: " << nMeshY << " nz: " << nMeshZ << std::endl; 

  std::vector<int> icells;
  for (int i=0; i<nParticle; ++i)
  {
      
    // Particle coordinates
    float xx0 = xx[i];
    float yy0 = yy[i];
    float zz0 = zz[i];
    float hh0 = hh[i];

    // Determine cell this particle sits in
    int ii0 = static_cast<int>(floor((xx0-x0)/dr));
    int jj0 = static_cast<int>(floor((yy0-y0)/dr));
    int kk0 = static_cast<int>(floor((zz0-z0)/dr));

    // Determine how many cells in each direction we must look
    int hcell = static_cast<int>(ceil(hh0/dr));

    // Clear array holding indices of chaining mesh cells this particle intersects
    icells.clear();
    int ncd = 2*hcell + 1;
    icells.reserve(ncd*ncd*ncd);

    // Add centre cell to the array
    int ncells = 0;
    icells.resize(1);
    icells[ncells] = CellIndex(ii0, jj0, kk0); 
    ncells++; 

    // Run through neihbouring cells and see if particle's smoothing sphere interscets
    for (int ii=ii0-hcell; ii<=ii0+hcell; ++ii) {
      if (ii < 0 || ii >= nMeshX) continue; 
      for (int jj=jj0-hcell; jj<=jj0+hcell; ++jj) {
        if (jj < 0 || jj >= nMeshY) continue;
        for (int kk=kk0-hcell; kk<=kk0+hcell; ++kk) {
          if (kk < 0 || kk >= nMeshZ) continue;
          if (ii == ii0 && jj == jj0 && kk == kk0) continue;
          bool intersects = CheckIntersection(xx0, yy0, zz0, hh0, ii, jj, kk); 
          if (intersects) {
            icells.resize(ncells + 1);
            icells[ncells] = CellIndex(ii, jj, kk);
            ncells++;
          }
        }
      }
    }

    // Add this particle's array index to each of the cells it intersects
    for (int ii=0; ii<ncells; ++ii) {
      int ic = icells[ii];
      std::vector<int> &cpi = ipi[ic];        
      int jj = cpi.size();
      cpi.resize(jj + 1);
      cpi[jj] = i;
     }

  }

  // Gather some stats on the list sizes
  int nmin = nParticle; 
  int nmax = 0;
  for (int i=0; i<nMesh; ++i) {
    std::vector<int> &cpi = ipi[i];
    int nc = cpi.size();
    nmin = std::min(nmin, nc);
    nmax = std::max(nmax, nc);
  }
  std::cout << " Chaining mesh setup complete ... nmin: " << nmin << " nmax: " << nmax << std::endl;
  std::cout << std::endl;

}

float ChainingMesh::ColumnDensity(float *xxs, float *yys, float *zzs, int nsamp)
{

  double colDensity = 0.0;

  for (int i=0; i<nsamp; ++i) {

    // Determine if interpolation point is within the chaining mesh bounds
    float xx0 = xxs[i];
    float yy0 = yys[i];
    float zz0 = zzs[i];
    bool inside = (xx0 >= x0 && xx0 < x1 && yy0 >= y0 && yy0 < y1 && zz0 >= z0 && zz0 < z1) ? true : false;

    if (inside) { 
      // Determine chaining mesh cell this point belongs to
      int ii0 = static_cast<int>(floor((xx0-x0)/dr));
      int jj0 = static_cast<int>(floor((yy0-y0)/dr));
      int kk0 = static_cast<int>(floor((zz0-z0)/dr));
      int ic0 = CellIndex(ii0, jj0, kk0);

      // Get interaction list for this cell 
      std::vector<int> &cpi = ipi[ic0];
      int np = cpi.size();

      // Cycle over all particles within the cell and increment value to colDensity
      for (int j=0; j<np; ++j) {
        int p = cpi[j];
        float xxp = xx[p];
        float yyp = yy[p];
        float zzp = zz[p];
        float hhp = hh[p];
        float vvp = vv[p]; 
        colDensity += SPHInterpolation(xx0, yy0, zz0, xxp, yyp, zzp, hhp, vvp);
      }
    
    }

  }

  return static_cast<float>(colDensity);

}

int ChainingMesh::CellIndex(int ii, int jj, int kk)
{
  // Returns the one-dimensional array index for the chaining mesh cell (ii, jj, kk)
  assert(ii >= 0 && ii < nMeshX && jj >= 0 && jj < nMeshY && kk >= 0 && kk < nMeshZ);
  return (ii*nMeshY + jj)*nMeshZ + kk;
}

bool ChainingMesh::CheckIntersection(float xxp, float yyp, float zzp, float hhp, int ii, int jj, int kk)
{

  // Extents of the chaining mesh cell
  float rxmin = x0 + ii*dr;
  float rxmax = rxmin + dr;
  float rymin = y0 + jj*dr;
  float rymax = rymin + dr;
  float rzmin = z0 + kk*dr;
  float rzmax = rzmin + dr;

  // Find distance to closest point on the mesh cell 
  // See: https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point 
  float h2 = hhp*hhp;
  float dx = std::max(rxmin-xxp, std::max(0.0f, xxp-rxmax));
  float dy = std::max(rymin-yyp, std::max(0.0f, yyp-rymax));
  float dz = std::max(rzmin-zzp, std::max(0.0f, zzp-rzmax));
  float r2 = dx*dx + dy*dy + dz*dz;

  return (r2 <= h2); 

}

double ChainingMesh::SPHInterpolation(float xxi, float yyi, float zzi, float xx0, float yy0, float zz0, float hh0, float vv0)
{

  // Compute relative distance
  double dx = xxi - xx0;
  double dy = yyi - yy0;
  double dz = zzi - zz0;
  double rr = sqrt(dx*dx + dy*dy + dz*dz);
  double eta = rr / hh0;
  
  // SPH interpolation weight
  double W = SPHKernel(eta, hh0); 

  // Return interpolated value
  return W*vv0;

}

double ChainingMesh::SPHKernel(double eta, double h)
{
  double ret = 0.0;
  double Anu = 4.9238560519; //(495.0f/32.0f/M_PI); 
  double r1 = (eta-1.0);
  double r2 = 1.0 + 6.0*eta + (35.0/3.0)*eta*eta;
  ret = r1*r1*r1*r1*r1*r1*r2;
  ret /= (h*h*h);
  ret *= Anu;
  return ret*(eta<ETAMAX);
}

ChainingMesh::~ChainingMesh()
{
  delete [ ] ipi; ipi = 0;
}

