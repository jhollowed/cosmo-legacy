// CloudsInCells.cxx 
//
// Class which splits the domain into cells and distributes particle masses among
// those cell vertices
//

#include "CloudsInCells.h"

CloudsInCells::CloudsInCells(int ptype, float xmin, float xmax, float ymin, float ymax, float zmin, 
                             float zmax, float dlen, int np, float *xxloc, float *yyloc, float *zzloc){

  // This function creates a mesh of cells, within which the CIC density estimatiion will
  // be performed, that spans the range specified with x,y,zmin,max
  //
  // Input parameters:
  // 
  // ptype: the particle type to be used in the projection (0 for dm, 1 for baryon). If
  //        ptype == 1, the function will stop and notify that in that casei, ChainingMesh
  //        should be used over CloudsInCells
  // x,y,zmin and x,y,zmax: the comoving cartesian domain of the final render in Mpc/h
  // dlen: symmetric width of mesh cells
  // np: number of particles in domain
  // xx,yy,zz: vectors containing particle position data
  
  if(ptype == 1){
      throw std::invalid_argument("If pttype==1, the ChainingMesh density estimation should be used");
  } 
    
  // Set mesh boundaries and determine how many cells per dimension there are
  x0 = xmin; x1 = xmax; Lx = xmax - xmin;
  y0 = ymin; y1 = ymax; Ly = ymax - ymin;
  z0 = zmin; z1 = zmax; Lz = zmax - zmin;
  dr = dlen;
  nMeshX = static_cast<int>(ceil(Lx/dr));
  nMeshY = static_cast<int>(ceil(Ly/dr));
  nMeshZ = static_cast<int>(ceil(Lz/dr));
  nMesh  = nMeshX*nMeshY*nMeshZ;
  
  // Instantiate two arrays of vectors, the vectors holding particle indices and 
  // densities for each cell
  cellParticles = new std::vector<int>[nMesh];
  cellDensity = new std::vector<float>[nMesh];
  
  // Set data arrays
  nParticle = np;
  xx = xxloc;
  yy = yyloc;
  zz = zzloc;

  // Fill cellParticles so that each array contains the list of all particles indices whose 
  // position is within the cell, and fll cellDensity so that each array contains the weighted
  // mass all particles whose position is within the cell.
  DistributeToCells();

}


void CloudsInCells::DistributeToCells()
{

  // This function fills all CIC cells with indices of their intersecting particles, 
  // and distributes the particles masses (normalized to 1) to cell verticies, weighted
  // inversely proportional to their distance from each vertex

  std::cout << " Setting up CIC mesh with nx: " << nMeshX << " ny: " << nMeshY << " nz: " << 
                                                nMeshZ << std::endl; 

  // loop through cells, checking  all particles to see if they should 
  // contribute to the cumulative density to that cell
  for (int i=0; i<nMeshX; ++i) 
  for (int j=0; j<nMeshX; ++j) 
  for (int k=0; k<nMeshX; ++k){ 
    
    // Determine position and index of cell's negative vertex at (x,y,z) = 
    // (-dr/2,-dr/2,-dr/2) with respect to the cell center
    float xcell = i * dr;
    float ycell = j * dr;
    float zcell = k * dr;
    int icell = CellIndex(i, j, k);

    for (int p=0; p<nParticle; ++p)
    {    
      // Particle coordinates
      float xx0 = xx[p];
      float yy0 = yy[p];
      float zz0 = zz[p];

      // Determine if this particle is within dr of this cells negative vertex
      if( abs(xx0 - xcell) > dr || 
          abs(yy0 - ycell) > dr ||
          abs(zz0 - zcell) > dr){
        continue;
      }

      // If so, then weight the particle's mass (normalized to 1) inversely to it's
      // distance from the vertex
      float xWeight = 1 - abs(xx0 - xcell) / dr;
      float yWeight = 1 - abs(yy0 - ycell) / dr;
      float zWeight = 1 - abs(zz0 - zcell) / dr;
      float pWeight = xWeight * yWeight * zWeight;

      // Get weighted density by dividing by cell volume
      // pWeight /= dr*dr*dr;

      // Add this particle's array index to this cell (in vector cellParticles) and add its 
      // density contribution to the cells cumulative density value (in vector cellDensity)--
      std::vector<int> &pIndices = cellParticles[icell];
      int isize = pIndices.size();
      pIndices.resize(isize + 1);
      pIndices[isize] = i;
      
      std::vector<float> &pDensities = cellDensity[icell];
      int dsize = pDensities.size();
      pDensities.resize(dsize + 1);
      pDensities[dsize] = pWeight;
    }  
  }

  // Gather some stats on the list sizes
  int nmin = nParticle; 
  int nmax = 0;
  for (int i=0; i<nMesh; ++i) {
    std::vector<int> &pIndices = cellParticles[i];
    int nc = pIndices.size();
    nmin = std::min(nmin, nc);
    nmax = std::max(nmax, nc);
  }
  std::cout << " CIC mesh setup complete ... " << std::endl; 
  std::cout << " min particles/cell: " << nmin << std::endl;
  std::cout << " max particles/cell: " << nmax << std::endl;
  std::cout << std::endl;

}


float CloudsInCells::ColumnDensity(float *xxs, float *yys, float *zzs, int nsamp)
{
 
 // Return the density of the cell column which the specified points (skewer) intersect
 //
 // Input args:
 // xxs, yys, zzs: x, y, and z positions defining the interpolation points 
 //                along the skewer, in Mpc/h
 // nsamp: the number of sampling points along the skewer (length of the xxs, yys, zzs vectors)

  double colDensity = 0.0;
  int prevCell = -1;

  // loop through all skewer points and get cell densities as calculated 
  // in DistributeToCells
  for (int i=0; i<nsamp; ++i) {

    // Determine if this interpolation point is within the CIC mesh bounds
    float xx0 = xxs[i];
    float yy0 = yys[i];
    float zz0 = zzs[i];
    bool inside = (xx0 >= x0 && xx0 < x1 && 
                   yy0 >= y0 && yy0 < y1 && 
                   zz0 >= z0 && zz0 < z1) ? true : false;

    if (inside) { 
      // Determine CIC mesh vertex-cell this point belongs to
      int ii0 = static_cast<int>(floor((xx0-x0)/dr));
      int jj0 = static_cast<int>(floor((yy0-y0)/dr));
      int kk0 = static_cast<int>(floor((zz0-z0)/dr));
      int thisCell = CellIndex(ii0, jj0, kk0);
     
      // if two or more sampling points fall in the same cell, only 
      // add density to column for one of them
      if (thisCell == prevCell){
        continue;
      } else {   
        prevCell = thisCell;
      }

      // Get particle index list for this cell 
      std::vector<int> &pIndices = cellParticles[thisCell];
      std::vector<float> &pDensities = cellDensity[thisCell];
      int np = pIndices.size();

      // Increment value to colDensity for each weighted particle in this cell
      for (int j=0; j<np; ++j) {
	colDensity += pDensities[j];
      }
    }
  }

  // done with all cells
  return static_cast<float>(colDensity);

}


int CloudsInCells::CellIndex(int ii, int jj, int kk)
{
  // Returns the one-dimensional array index for the chaining mesh cell (ii, jj, kk)
  assert(ii >= 0 && ii < nMeshX && jj >= 0 && jj < nMeshY && kk >= 0 && kk < nMeshZ);
  return (ii*nMeshY + jj)*nMeshZ + kk;
}


CloudsInCells::~CloudsInCells()
{
  delete [ ] cellParticles; cellParticles = 0;
}

