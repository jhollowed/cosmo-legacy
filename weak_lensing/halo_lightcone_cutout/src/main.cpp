#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <stdexcept>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <omp.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <vector>

// Generic IO
#include "GenericIO.h"
//#include "GenericIOPosixReader.h"

// Cosmotools
#define REAL double
/************* LENSING HEADERS ***************/
//#include "mycosmology.h"
//#include "lensing_funcs.h"

#define PI 3.14159265
#define ARCSEC 3600.0

using namespace std;
using namespace gio;

struct Buffers {
  vector<float> x;
  vector<float> y;
  vector<float> z;
  vector<float> vx;
  vector<float> vy;
  vector<float> vz;
  vector<float> a;
  vector<int>   step;
  vector<int64_t> id;
  vector<float> theta;
  vector<float> phi;
};



// LC fields
//x     y       z       vx      vy      vz      phi     id      a       mask    mass    step

int getdir (string dir, vector<string> &files) {
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
    cout << "Error(" << errno << ") opening " << dir << endl;
    return errno;
  }

  while ((dirp = readdir(dp)) != NULL) {
    if (string(dirp->d_name).find("lc_output")!=string::npos) {
      files.push_back(string(dirp->d_name));
    }
  }
  closedir(dp);
  return 0;
}

float redshift(float a) {
  return 1.0f/a-1.0f;
}

void processLC(string dir_name, string out_dir, vector<string> step_strings) {

  Buffers b;

  cout << "Reading directory:" << dir_name << endl;
  vector<string> files;
  getdir(dir_name,files);

  size_t max_size = 0;
  for (int i=0; i<step_strings.size();++i){
      ostringstream file_name;
      std::cout<<step_strings[i]<<std::endl;
      //      file_name << dir_name << "/lc_output." << step_strings[i];
      file_name << dir_name << "/lc" << step_strings[i]<<"/glc";

      cout << "Opening file:" << file_name.str() << endl;
      // GenericIOPosixReader *reader = new GenericIOPosixReader();
      // reader->SetFileName(file_name.str());
      // reader->SetCommunicator(MPI_COMM_SELF);
      GenericIO reader(MPI_COMM_SELF,file_name.str());
      reader.openAndReadHeader(GenericIO::MismatchRedistribute);
      int nRanks = reader.readNRanks();
      for (int j=0; j<nRanks; ++j) {
        size_t current_size = reader.readNumElems(j);
        max_size = current_size > max_size ? current_size : max_size;
      }
      max_size +=10;
      std::cout<<"max size: "<<max_size<<std::endl;
      b.x.resize(max_size);
      b.y.resize(max_size);
      b.z.resize(max_size);
      b.vx.resize(max_size);
      b.vy.resize(max_size);
      b.vz.resize(max_size);

      b.a.resize(max_size);
      //b.step.resize(max_size);
      b.id.resize(max_size);
      b.theta.resize(max_size);
      b.phi.resize(max_size);
      std::cout<<"done resizing"<<std::endl;
      int step =atoi(step_strings[i].c_str());
      // std::size_t found = file_name.str().find_last_of(".");
      // step = atoi(file_name.str().substr(found+1).c_str());

      ofstream id_file;
      ofstream theta_file;
      ofstream phi_file;
      ofstream redshift_file;
      ofstream x_file;
      ofstream y_file;
      ofstream z_file;
      ofstream vx_file;
      ofstream vy_file;
      ofstream vz_file;


      ostringstream id_file_name;
      ostringstream theta_file_name;
      ostringstream phi_file_name;
      ostringstream redshift_file_name;
      ostringstream x_file_name;
      ostringstream y_file_name;
      ostringstream z_file_name;
      ostringstream vx_file_name;
      ostringstream vy_file_name;
      ostringstream vz_file_name;

      id_file_name << out_dir << "/id." << step << ".bin";
      theta_file_name << out_dir << "/theta." << step << ".bin";
      phi_file_name << out_dir << "/phi." << step << ".bin";
      redshift_file_name << out_dir << "/redshift." << step << ".bin";
      x_file_name << out_dir << "/x."<<step<<".bin";
      y_file_name << out_dir << "/y."<<step<<".bin";
      z_file_name << out_dir << "/z."<<step<<".bin";
      vx_file_name << out_dir << "/vx."<<step<<".bin";
      vy_file_name << out_dir << "/vy."<<step<<".bin";
      vz_file_name << out_dir << "/vz."<<step<<".bin";
      std::cout<<"starting to open file"<<std::endl;
      id_file.open(id_file_name.str().c_str(), ios::out | ios::binary);
      theta_file.open(theta_file_name.str().c_str(), ios::out | ios::binary);
      phi_file.open(phi_file_name.str().c_str(), ios::out | ios::binary);
      redshift_file.open(redshift_file_name.str().c_str(), ios::out | ios::binary);
      x_file.open(x_file_name.str().c_str(), ios::out | ios::binary);
      y_file.open(y_file_name.str().c_str(), ios::out | ios::binary);
      z_file.open(z_file_name.str().c_str(), ios::out | ios::binary);
      vx_file.open(vx_file_name.str().c_str(), ios::out | ios::binary);
      vy_file.open(vy_file_name.str().c_str(), ios::out | ios::binary);
      vz_file.open(vz_file_name.str().c_str(), ios::out | ios::binary);
      std::cout<<"done opening files"<<std::endl;
      reader.addVariable("x", &b.x[0]); 
      reader.addVariable("y", &b.y[0]); 
      reader.addVariable("z", &b.z[0]); 
      reader.addVariable("vx", &b.vx[0]); 
      reader.addVariable("vy", &b.vy[0]); 
      reader.addVariable("vz", &b.vz[0]); 
      reader.addVariable("a", &b.a[0]); 
      //reader.addVariable("step", &b.step[0]); 
      reader.addVariable("id", &b.id[0]); 

      for (int j=0; j<nRanks; ++j) {
        size_t current_size = reader.readNumElems(j);
        cout << "Reading:" << current_size << endl;
        reader.readData(j);
	
	std::cout << "Converting positions..."<<j+1<<"/"<<nRanks << std::endl;
        for (int k=0; k<current_size; ++k) {
          if (b.x[k] > 0.0 && b.y[k] > 0.0 && b.z[k] > 0.0) {
            float r = (float)sqrt(b.x[k]*b.x[k]+b.y[k]*b.y[k]+b.z[k]*b.z[k]);
            b.theta[k] = acos(b.z[k]/r) * 180.0 / PI * ARCSEC;
            b.phi[k]   = atan(b.y[k]/b.x[k]) * 180.0 / PI * ARCSEC;
            if (b.theta[k] > theta_cut[0] && b.theta[k] < theta_cut[1] && 
                b.phi[k] > phi_cut[0] && b.phi[k] < phi_cut[1] ) {
	      float tmp = redshift(b.a[k]);
	      id_file.write( (char*)&b.id[k], sizeof(int64_t));
	      theta_file.write( (char*)&b.theta[k], sizeof(float));
	      phi_file.write( (char*)&b.phi[k], sizeof(float));
	      x_file.write((char*)&b.x[k],sizeof(float));
	      y_file.write((char*)&b.y[k],sizeof(float));
	      z_file.write((char*)&b.z[k],sizeof(float));
	      vx_file.write((char*)&b.vx[k],sizeof(float));
	      vy_file.write((char*)&b.vy[k],sizeof(float));
	      vz_file.write((char*)&b.vz[k],sizeof(float));
	      redshift_file.write( (char*)&tmp, sizeof(float));
            }
          }
        }
      }
      
      reader.close();
      id_file.close();
      theta_file.close();
      phi_file.close();
      x_file.close();
      y_file.close();
      z_file.close();
      vx_file.close();
      vy_file.close();
      vz_file.close();
      redshift_file.close();
  }
}


int main( int argc, char** argv ) {

  // This code generates a cutout from a large lightcone run by finding all
  // particles/objects residing within a volume defined by theta and phi bounds 
  // in spherical coordaintes, centered on the "observer" (at the origin, by default)
  //
  // Three mandatory input arguments are required:
  // - path to the input lightcone
  // - output path
  // - steps to use 
  //
  // Further arguments are optional depending on your use case. 
  // new inputs needed: center theta & phi & rho (spherical coords), box side length
  
  MPI_Init(&argc, &argv);

  string input_lc_dir,out_dir;
  input_lc_dir = string(argv[1]);
  out_dir = string(argv[2]);
  vector<string> step_strings;
  std::cout << "using lightcone at ";
  std::cout << input_lc_dir << std::endl;
   
  // build step_strings vector out of all command line arguments after the input
  // and output file paths, until end of argv, or another input option is found (next 
  // element of argv cannot be interpreted as an int). 
  char* p;
  bool isInt;
  int lastStep_idx;
  std::cout << "step strings: ";
  for(int i = 3; i<argc; ++i){
    strtol(argv[i], &p, 10);
    isInt = (*p == 0);
    if(isInt){
    	step_strings.push_back(string(argv[i]));
	std::cout << string(argv[i]);
    }else{
	lastStep_idx = i;
	break;
    }
  }
  std::cout << std::endl;

  // set parameter defaults
  float observer[3] = {0.0, 0.0, 0.0};
  float theta_cut[2] = {85.0*ARCSEC, 90.0*ARCSEC};
  float phi_cut[2]   = {0.0,         5.0 *ARCSEC};
  double haloPos[3];
  float boxLength;

  // check that supplied arguments are valid
  vector<string> args(argv+1, argv + argc);
  bool customThetaBounds = (std::find(args.begin(), args.end(), "-t") != args.end()) ||
  		     	   (std::find(args.begin(), args.end(), "--theta") != args.end());
  bool customPhiBounds = (std::find(args.begin(), args.end(), "-p") != args.end()) ||
  		   	 (std::find(args.begin(), args.end(), "--phi") != args.end());
  bool customHalo = (std::find(args.begin(), args.end(), "-h") != args.end()) ||
  		    (std::find(args.begin(), args.end(), "--halo") != args.end());
  bool customBox = (std::find(args.begin(), args.end(), "-b") != args.end()) ||
  		   (std::find(args.begin(), args.end(), "--boxLength") != args.end());
  
  // there are two general use cases of this cutout code, as described in the doc string 
  // below the declaration of this main function. Here, exceptons are thrown to prevent 
  // confused input arguments which mix those two use cases.
  if(customHalo^customBox){ 
	throw std::invalid_argument("-h and -b options must accompany eachother");
  }
  if(customHalo && (customThetaBounds || customPhiBounds)){
  	throw std::invalid_argument("-t and -p options must not be used in the case " \
                                    "that -h and -b arguments are passed");
  }

  // search argument vector for options, update default parameters if found 
  for(int i=lastStep_idx; i<argc; ++i){

    if(strcmp(argv[i],"-o")==0 || strcmp(argv[i],"--observer")==0){
	observer[0] = std::strtof(argv[++i], NULL);
	observer[1] = std::strtof(argv[++i], NULL);
	observer[2] = std::strtof(argv[++i], NULL);
    }
    else if(strcmp(argv[i],"-t")==0 || strcmp(argv[i],"--theta")==0){
	theta_cut[0] = std::strtof(argv[++i], NULL) * ARCSEC;
	theta_cut[1] = std::strtof(argv[++i], NULL) * ARCSEC;
    }
    else if(strcmp(argv[i],"-p")==0 || strcmp(argv[i],"--phi")==0){
	phi_cut[0] = std::strtof(argv[++i], NULL) * ARCSEC;
	phi_cut[1] = std::strtof(argv[++i], NULL) * ARCSEC;
    }
    else if(strcmp(argv[i],"-h")==0 || strcmp(argv[i],"--halo")==0){
	customHalo = 1;
	haloPos[0] = std::strtof(argv[++i], NULL);
	haloPos[1] = std::strtof(argv[++i], NULL);
	haloPos[2] = std::strtof(argv[++i], NULL);	
    }
    else if (strcmp(argv[i],"-b")==0 || strcmp(argv[i],"--boxLength")==0){
	customBox = 1;
	boxLength = std::strtof(argv[++i], NULL);
    }
  }

  // A box size must be passed if pointing lightcone toward custom halo position,
  // in order to set the angular bounds
  if(customHalo^customBox){ 
	throw std::invalid_argument("-h and -b options must accompany eachother");
  }
  
  // call overloaded processing function
  if(customHalo){
  	//processLC(input_lc_dir, out_dir, step_strings, observer, haloPos, boxLength);
  }else{
	//processLC(input_lc_dir, out_dir, step_strings, observer, theta_cut, phi_cut);
  }
  MPI_Finalize();
  return 0;
}
