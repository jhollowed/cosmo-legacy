// Skewer.cxx 
//
// Class that constructs interpolation skewers based on an observers position
// and pixel plane.
//

#include "Skewer.h"
#include <iostream>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265f
#endif

Skewer::Skewer(float xobs, float yobs, float zobs, float xrot, float yrot, float zrot, float zdepth, 
               float dfovx, float dfobs, float dtheta, float dphi, int nx, int ny, int nz, int nthreads) 
{

  // Observer location
  xo = xobs;
  yo = yobs;
  zo = zobs;

  // Rotation point
  xr = xrot;
  yr = yrot;
  zr = zrot;

  // Convert degrees to radians 
  float xfov = dfovx * M_PI / 180.0f;
  theta = dtheta * M_PI / 180.0f;
  phi   = dphi * M_PI / 180.0f;
  pivot = 0.0f;

  // Pixel plane width and z location 
  float dpx = 2*zdepth*tan(xfov/2.);
  float dpy = dpx*ny/(1.0f*nx); 
  zp = zobs + zdepth;
   
  // Pixel separation
  dx = dpx / nx;
  dy = dpy / ny;

  // Centre of origin pixels
  xp = xo - dpx/2 + dx/2;
  yp = yo - dpy/2 + dy/2;

  // Ray separation on observer plane
  dx0 = dfobs*dx;
  dy0 = dfobs*dy;

  // Origin of rays on observer plane
  xo = xo - dfobs*dpx/2 + dx0/2;
  yo = yo - dfobs*dpy/2 + dy0/2;  

  // Number of interpolation points along the skewer
  nsamp = nz;

  // Instantiate arrays holding sample points for each thread
  xxs = new std::vector<float>[nthreads];
  yys = new std::vector<float>[nthreads];
  zzs = new std::vector<float>[nthreads];
  for (int i=0; i<nthreads; ++i) {
    std::vector<float> &xxt = xxs[i]; xxt.resize(nsamp);
    std::vector<float> &yyt = yys[i]; yyt.resize(nsamp);
    std::vector<float> &zzt = zzs[i]; zzt.resize(nsamp);
  }

  // Print some information
  std::cout << " Pixel plane is located at zp : " << zp << std::endl;
  std::cout << " Pixel plane width is         : " << "( " << dpx << " , " << dpy << " ) " << std::endl;
  std::cout << " Fraction of observer plane   : " << dfobs << std::endl; 
  std::cout << " Pixel separation on plane is : " << dx << " " << dy << std::endl;
  std::cout << " Lower-left coordinate is     : " << " ( " << xp << " , " << yp << " ) " << std::endl;
  std::cout << " Upper-right coordinate is    : " << " ( " << xp+dx*(nx-1) << " , " << yp+dy*(ny-1) << " ) " << std::endl;
  std::cout << " Rotation origin              : " << " ( " << xr << " , " << yr << " , " << zr << " ) " << std::endl;
  std::cout << " Rotation in yz plane         : " << dtheta << std::endl;
  std::cout << " Rotation in xy plane         : " << dphi << std::endl;
  std::cout << " Openmp threads               : " << nthreads << std::endl;
  std::cout << std::endl;

}

void Skewer::InterpolationPoints(int nt, int ipix, int jpix)
{

  // Starting and stopping points relative to the rotation origin
  float x0_ = xo + ipix*dx0 - xr;
  float y0_ = yo + jpix*dy0 - yr;
  float z0_ = zo - zr; 
  float x1_ = xp + ipix*dx - xr;
  float y1_ = yp + jpix*dy - yr;
  float z1_ = zp - zr;

  // Apply rotations (theta is in yz plane; phi is in xz plane)
#ifndef USE_PHI_IN_XZ
  float x0 = x0_*cos(phi) - y0_*sin(theta)*sin(phi) - z0_*cos(theta)*sin(phi);
  float y0 = y0_*cos(theta) - z0_*sin(theta);
  float z0 = x0_*sin(phi) + y0_*sin(theta)*cos(phi) + z0_*cos(theta)*cos(phi);
  float x1 = x1_*cos(phi) - y1_*sin(theta)*sin(phi) - z1_*cos(theta)*sin(phi);
  float y1 = y1_*cos(theta) - z1_*sin(theta);
  float z1 = x1_*sin(phi) + y1_*sin(theta)*cos(phi) + z1_*cos(theta)*cos(phi);
#else
  float x0 = x0_*cos(phi) - y0_*sin(phi);
  float y0 = x0_*cos(theta)*sin(phi) + y0_*cos(theta)*cos(phi) - z0_*sin(theta);
  float z0 = x0_*sin(theta)*sin(phi) + y0_*sin(theta)*cos(phi) + z0_*cos(theta);
  float x1 = x1_*cos(phi) - y1_*sin(phi);
  float y1 = x1_*cos(theta)*sin(phi) + y1_*cos(theta)*cos(phi) - z1_*sin(theta);
  float z1 = x1_*sin(theta)*sin(phi) + y1_*sin(theta)*cos(phi) + z1_*cos(theta);
#endif

  // Convert back to absolute frame
  x0 += xr;
  y0 += yr;
  z0 += zr;
  x1 += xr;
  y1 += yr;
  z1 += zr;

  // Apply pivot (negative of pivot really due to how colormap works)
  x0 -= pivot;
  x1 += pivot;

  // Interpolation widths
  float dxi = (x1 - x0) / nsamp;
  float dyi = (y1 - y0) / nsamp;
  float dzi = (z1 - z0) / nsamp; 

  // Find interpolation points starting wtih x0 + dxi (don't sample the origin)
  std::vector<float> &xxt = xxs[nt];
  std::vector<float> &yyt = yys[nt];
  std::vector<float> &zzt = zzs[nt];
  for (int i=0; i<nsamp; ++i) {
    xxt[i] = x0 + dxi*(i+1);
    yyt[i] = y0 + dyi*(i+1);
    zzt[i] = z0 + dzi*(i+1);
  }

}

Skewer::~Skewer()
{
  delete [ ] xxs; xxs = 0;
  delete [ ] yys; yys = 0;
  delete [ ] zzs; zzs = 0;
}

