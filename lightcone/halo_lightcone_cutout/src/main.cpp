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
#include <numeric>
#include <algorithm>
#include <sstream>
#include <omp.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <vector>

// Generic IO
#include "GenericIO.h"

// Cosmotools
#define REAL double

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
    vector<int>     step;
    vector<int64_t> id;
    vector<float> theta;
    vector<float> phi;
};

int getdir (string dir, vector<string> &files) {
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
    cout << "Error(" << errno << ") opening data files" << dir << endl;
    return errno;
    }
    while ((dirp = readdir(dp)) != NULL) {
    files.push_back(string(dirp->d_name));
    //if (string(dirp->d_name).find("lc_output")!=string::npos) {
    //#}
    }
    closedir(dp);
    return 0;
}

float redshift(float a) {
    return 1.0f/a-1.0f;
}

vector<float> cross(const vector<float> & v1, 
                    const vector<float> & v2){
    // This function returns the cross product of the vectors v1 and v2
    
    auto n = v1.size();
    auto n2 = v2.size();
    assert(("vectors v1 and v2 must have the same length", n == n2));

    vector<float> vv(3);
    vv[0] = v1[1]*v2[2] - v1[2]*v2[1];
    vv[1] = -(v1[0]*v2[2] - v1[2]*v2[0]);
    vv[2] = v1[0]*v2[1] - v1[1]*v2[0];
    return vv;

}

void rotate(const vector<float> & a, 
            const vector<float> & b, 
            const float B, 
            const vector<float> & v_vec, 
            vector<float> & v_rot){ 
    // This function implements the Rodrigues rotation formula. a and
    // b are two vectors defining the plane of rotation, and thus allow
    // calculating the axis of rotation k. B is the angle of rotation, and
    // v_vec is a cartesian position vector to be rotated an angle B about k.
    // The rotated vector is stored in the vector v_rot passed by the user.

    // MOVE CALCULATION OF K TO NEW FUNCTION

    int na = a.size();
    int nb = b.size();
    int nv = v_vec.size();
    assert(("vectors a, b, and v must have the same length", na==nb && na==nv));
    
    // find k
    vector<float> axb = cross(a, b);
    float mag_axb = sqrt(std::inner_product(axb.begin(), axb.end(), axb.begin(), 0.0));
    vector<float> k_vec(3);
    for(int i=0; i<na; ++i){ k_vec[i] = axb[i] / mag_axb; }
    
    // find (k ⨯ v) and (k·v)
    vector<float> kxv_vec = cross(k_vec, v_vec);
    float kdv = std::inner_product(k_vec.begin(), k_vec.end(), v_vec.begin(), 0.0);
    
    // do rotation
    float v, k, k_x_v;
    for(int i=0; i<na; ++i){
        v = v_vec[i];
        k = k_vec[i];
        k_x_v = kxv_vec[i];

        v_rot[i] = v*cos(B) + (k_x_v)*sin(B) + k*kdv*(1-cos(B));
    }
      
}


void processLC(string dir_name, string out_dir, vector<string> step_strings) {

    Buffers b;

    cout << "Reading directory:" << dir_name << endl;
    vector<string> files;
    getdir(dir_name, files);
    for (vector<string>::const_iterator i = files.begin(); i != files.end(); ++i)
         cout << *i << ' '; 

    size_t max_size = 0;
    for (int i=0; i<step_strings.size();++i){
        ostringstream file_name;
        cout<<step_strings[i]<<endl;
        file_name << dir_name << "/lc" << step_strings[i] << "/";

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
        cout<<"max size: "<<max_size<<endl;
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
        cout<<"done resizing"<<endl;
        int step =atoi(step_strings[i].c_str());
        // size_t found = file_name.str().find_last_of(".");
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
        cout<<"starting to open file"<<endl;
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
        cout<<"done opening files"<<endl;
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
    
    cout << "Converting positions..."<<j+1<<"/"<<nRanks << endl;
        for (int k=0; k<current_size; ++k) {
            if (b.x[k] > 0.0 && b.y[k] > 0.0 && b.z[k] > 0.0) {
            float r = (float)sqrt(b.x[k]*b.x[k]+b.y[k]*b.y[k]+b.z[k]*b.z[k]);
            b.theta[k] = acos(b.z[k]/r) * 180.0 / PI * ARCSEC;
            b.phi[k]     = atan(b.y[k]/b.x[k]) * 180.0 / PI * ARCSEC;
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

    // This code generates a cutout from a larger lightcone run by finding all
    // particles/objects residing within a volume defined by ɵ and ϕ bounds 
    // in spherical coordaintes, centered on the observer (assumed to be the 
    // spatial origin)
    //
    // Two use cases are supported: 
    //
    /////////////////////////////////////////
    //
    // Define the ɵ and ϕ bounds explicitly:
    // 
    // lc_cutout <input lightcone dir> <output dir> <max redshift> --theta <ɵ_center> <dɵ> --phi <ϕ_center> <dϕ>
    // 
    // where the ϕ_center argument is the azimuthal coordinate of the center of the 
    // field of view that one wishes to cut out of the lightcone, and dϕ is the 
    // angualar distance from this center to the edge of the cutout and likewise 
    // for the similar ɵ args. That is, the result will be a sky area that spans 
    // 
    // (ɵ_center - dɵ)  ≤  ɵ    ≤  (ɵ_center + dɵ)
    // (ϕ_center - dϕ)  ≤  ɵ    ≤  (ϕ_center + dϕ)
    //
    // The expected units are DEGREES. The --theta and --phi flags can be replaced 
    // with -t and -p
    //
    /////////////////////////////////////////
    //
    // Allow the ɵ and ϕ bounds to be computed interanally to obtain a cutout of 
    // a certain width, in Mpc/h, centered on a certain cartesian positon, 
    // (x, y, z) Mpc/h (intended to be used for cutting out cluster-sized halos):
    //
    // lc_cutout <input lightcone dir> <output dir> <max redshift> --halo <x> <y> <z> --boxLength <box length>
    //
    // The --halo and --boxLength flags can be replaced with -h and -b
    //
    // We want to express the positions of  all of our lightcone objects in 
    // spherical coordinates, to perform the cutout, and we want that coordinate 
    // system to be rotated such that the halo of intererst lies on the equator at
    // 
    // (r_halo, 90°, 0°)
    // where r_halo = (x^2 + y^2 + z^2)^(1/2)
    // 
    // Let's call the position vector of the halo before this rotation 
    // a = [x, y, z], and after, b = [x_rot, y_rot, z_rot] = [r_halo, 0, 0]
    //
    // We perform this rotation for each lightcone object via the Rodrigues rotation
    // formula, which answers the following question: given a position vector v, a 
    // normalized axis of rotation k, and an angle of rotation β, what is an 
    // analytical form for a new vector v_rot which is v rotated by an anlge β 
    // about k?
    //
    // First, we find k by taking the cross product of two vectors defining the 
    // plane of rotation. The obvious choice of these two vectors are a and b, as 
    // defined above;
    //
    // k = (a ⨯ b) / ||a ⨯ b||
    //
    // then, for any other position vector v, v_rot is given by
    //
    // v_rot = vcosβ + (k ⨯ v)sinβ + k(k·v)(1-cosβ)
    //
    // This coordinate rotation is required because the bounding functions which 
    // define the field of view of the observer, while constant in theta-phi space, 
    // are nonlinear in cartesian space. The distortion is maximized near the poles 
    // of the spherical coordinate system, and minmized at the equator. Areas 
    // defined by constant theta-phi bounds then appear trapezoidal to the observer 
    // when far from the equator. It is important that our cutout areas are 
    // maintained as square for at least two reasons:
    //
    // - At the moment, fft restrictions of flat-sky lensing code require that the 
    //   cutout is square
    // - The cutouts returned will not actually have all side lengths of boxLength 
    //   if we don't do this rotation, which the user explicitly requested
    //
    ////////////////////////////////////////
    //
    // Note that the firt use case describe does not perform the coordinate 
    // rotation which is described in the second. So, cutouts returned will
    // not necessarily be square, or symmetrical.
    
    MPI_Init(&argc, &argv);

    char cart[3] = {'x', 'y', 'z'};
    string input_lc_dir, out_dir;
    input_lc_dir = string(argv[1]);
    out_dir = string(argv[2]);
    vector<string> step_strings;
    cout << "\nusing lightcone at ";
    cout << input_lc_dir << endl;
     
    // build step_strings vector out of all command line arguments after the input
    // and output file paths, until end of argv, or another input option is found (next 
    // element of argv cannot be interpreted as an int). 
    char* p;
    bool isInt;
    int lastStep_idx;
    cout << "steps to include: ";
    for(int i = 3; i<argc; ++i){
    strtol(argv[i], &p, 10);
    isInt = (*p == 0);
    if(isInt){
        step_strings.push_back(string(argv[i]));
        cout << string(argv[i]) << " ";
    }else{
        lastStep_idx = i;
        break;
    }
    }
    cout << endl;

    vector<float> theta_cut(2);
    vector<float > phi_cut(2);
    vector<float> haloPos(3);
    float boxLength;

    // check that supplied arguments are valid
    vector<string> args(argv+1, argv + argc);
    bool customThetaBounds = int((find(args.begin(), args.end(), "-t") != args.end()) ||
                             (find(args.begin(), args.end(), "--theta") != args.end()));
    bool customPhiBounds = int((find(args.begin(), args.end(), "-p") != args.end()) ||
                           (find(args.begin(), args.end(), "--phi") != args.end()));
    bool customHalo = int((find(args.begin(), args.end(), "-h") != args.end()) ||
                      (find(args.begin(), args.end(), "--halo") != args.end()));
    bool customBox = int((find(args.begin(), args.end(), "-b") != args.end()) ||
                     (find(args.begin(), args.end(), "--boxLength") != args.end()));
    
    // there are two general use cases of this cutout code, as described in the doc string 
    // below the declaration of this main function. Here, exceptons are thrown to prevent 
    // confused input arguments which mix those two use cases.
    if(customHalo^customBox){ 
        throw invalid_argument("-h and -b options must accompany eachother");
    }
    if(customThetaBounds^customPhiBounds){
        throw invalid_argument("-t and -p options must accompany eachother");
    }
    if(customHalo & customThetaBounds){
        throw invalid_argument("-t and -p options must not be used in the case " \
                                    "that -h and -b arguments are passed");
    }
    if(!customThetaBounds && !customPhiBounds && !customHalo && !customBox){
        throw invalid_argument("Valid options are -h, -b, -t, and -p")
    }

    // search argument vector for options, update default parameters if found
    // Note to future self: strcmp() returns 0 if the input strings are equal. See docs
    // for more info
    for(int i=lastStep_idx; i<argc; ++i){

        if(strcmp(argv[i],"-t")==0 || strcmp(argv[i],"--theta")==0){
            float theta_center = strtof(argv[++i], NULL) * ARCSEC;
            float dtheta = strtof(argv[++i], NULL) * ARCSEC;
            theta_cut[0] = theta_center - dtheta;
            theta_cut[1] = theta_center + dtheta;
        }
        else if(strcmp(argv[i],"-p")==0 || strcmp(argv[i],"--phi")==0){
            float phi_center = strtof(argv[++i], NULL) * ARCSEC;
            float dphi  = strtof(argv[++i], NULL) * ARCSEC;
            phi_cut[0] = phi_center - dphi;
            phi_cut[1] = phi_center + dphi;
        }
        else if(strcmp(argv[i],"-h")==0 || strcmp(argv[i],"--halo")==0){
            haloPos[0] = strtof(argv[++i], NULL);
            haloPos[1] = strtof(argv[++i], NULL);
            haloPos[2] = strtof(argv[++i], NULL);  
        }
        else if (strcmp(argv[i],"-b")==0 || strcmp(argv[i],"--boxLength")==0){
            boxLength = strtof(argv[++i], NULL);
        }
    }

    if(customHalo){
        cout << "target halo: ";
        for(int i=0;i<3;++i){ cout << cart[i] << "=" << haloPos[i] << " ";}
        cout << endl;
        cout << "box length: " << boxLength << " Mpc";
    }else{
        cout << "theta bounds: ";
        cout << theta_cut[0]/ARCSEC << " -> " << theta_cut[1]/ARCSEC <<" deg"<<endl;
        cout << "phi bounds: ";
        cout << phi_cut[0]/ARCSEC << " -> " << phi_cut[1]/ARCSEC <<" deg";
    }
    cout << "\n" << endl;
        
    // call overloaded processing function
    if(customHalo){
        processLC(input_lc_dir, out_dir, step_strings, haloPos, boxLength);
    }else{
        processLC(input_lc_dir, out_dir, step_strings, theta_cut, phi_cut);
    }

    MPI_Finalize();
    return 0;
}
