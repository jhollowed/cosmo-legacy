// Skewer.h 
//
// Class that constructs interpolation skewers based on an observers position
// and pixel plane.
//

#ifndef Skewer_h 
#define Skewer_h 

#include <vector>
#include <iostream>
#include <math.h>

class Skewer
  {

    public:

      Skewer(float xobs, float yobs, float zobs, float xrot, float yrot, float zrot, float zdepth, 
             float dfovx, float dfobs, float dtheta, float dphi, int nx, int ny, int nz, int nthreads);
      ~Skewer();

      void InterpolationPoints(int nt, int ipix, int jpix);

      float* ExtractX(int nt) { std::vector<float> &xxt = xxs[nt]; return &xxt[0]; }
      float* ExtractY(int nt) { std::vector<float> &yyt = yys[nt]; return &yyt[0]; }
      float* ExtractZ(int nt) { std::vector<float> &zzt = zzs[nt]; return &zzt[0]; }

    protected: 

      // Observer location
      float xo, yo, zo;

      // Point about which to rotate
      float xr, yr, zr;

      // Origin of pixel plane
      float xp, yp, zp;

      // Pixel separtion on the pixel plane 
      float dx, dy;

      // Ray separation on observer plan
      float dx0, dy0; 

      // Rotation in yz plane (theta) and xy plane (phi)
      float theta, phi;

      // Pivot in xy plane used to change vantage point
      float pivot;

      // Number of interpolation points along the skewer
      int nsamp;

      // Store sample points for each thread 
      std::vector<float> *xxs;
      std::vector<float> *zzs;
      std::vector<float> *yys;

  } ;
#endif

