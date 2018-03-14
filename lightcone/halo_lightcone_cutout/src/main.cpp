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

//////////////////////////////////////////////////////

struct Buffers {
    
    // From LC output
    vector<float> x;
    vector<float> y;
    vector<float> z;
    vector<float> vx;
    vector<float> vy;
    vector<float> vz;
    vector<float> a;
    vector<int> step;
    vector<int64_t> id;
    vector<int> rotation;
    vector<int32_t> replication;
    // New data columns
    vector<float> theta;
    vector<float> phi;
};

//////////////////////////////////////////////////////
//
//         Helper Functions
//
//////////////////////////////////////////////////////

float redshift(float a) {
    // Converts scale factor to redshift.
    //
    // Params:
    // :param a: the scale factor
    // :return: the redshift corresponding to input a
    
    return (1.0f/a)-1.0f;
}


float zToStep(float z, int totSteps=499, float maxZ=200.0){
    // Function to convert a redshift to a step number, rounding 
    // toward a = 0.
    //
    // Params:
    // :param z: the input redshift
    // :totSteps: the total number of steps of the simulation of 
    //            interest. Note-- the initial conditions are not 
    //            a step! totSteps should be the maximum snapshot 
    //            number.
    // :maxZ: the initial redshift of the simulation
    // :return: the simulation step corresponding to the input redshift, 
    //          rounded toward a = 0

    float amin = 1/(maxZ + 1);
    float amax = 1.0;
    float adiff = (amax-amin)/(totSteps-1);
    
    float a = 1/(1+z);
    cout << adiff << endl;
    int step = floor((a-amin) / adiff);
    return step;
}

//////////////////////////////////////////////////////
//
//         Coord Rotation functions
//
//////////////////////////////////////////////////////

void cross(const vector<float> &v1, 
           const vector<float> &v2,
           vector<float> &v1xv2){
    // This function calculates the cross product of two three 
    // dimensional vectors
    //
    // Parmas:
    // :param v1: some three-dimensional vector
    // :param v2: some other three-dimensional vector
    // :param v1xv2: vector to hold the resultant cross-product of 
    //               vectors v1 and v2
    // :return: none
    
    int n1 = v1.size();
    int n2 = v2.size();
    assert(("vectors v1 and v2 must have the same length", n1 == n2));

    v1xv2[0] = v1[1]*v2[2] - v1[2]*v2[1];
    v1xv2[1] = -(v1[0]*v2[2] - v1[2]*v2[0]);
    v1xv2[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


void normCross(const vector<float> &a,
           const vector<float> &b,
           vector<float> &k){
    // This function returns the normalized cross product of two three-dimensional
    // vectors. The notation, here, is chosen to match that of the Rodrigues rotation 
    // formula for the rotation vector k, rather than matching the notation of cross() 
    // above. Feel free to contact me with urgency if you find this issue troublesome.
    //
    // Parms:
    // :param a: some three-dimensional vector
    // :param b: some other three-dimensional vector
    // :param k: vector to hold the resultant normalized cross-product of vectors a and b
    // :return: none

    int na = a.size();
    int nb = b.size();
    assert(("vectors a and b must have the same length", na==nb));

    vector<float> axb(3); 
    cross(a, b, axb);
    float mag_axb = sqrt(std::inner_product(axb.begin(), axb.end(), axb.begin(), 0.0));
    for(int i=0; i<na; ++i){ k[i] = axb[i] / mag_axb; }
}


void rotate(const vector<float> &k_vec,
            const float B, 
            const vector<float> &v_vec, 
            vector<float> &v_rot){ 
    // This function implements the Rodrigues rotation formula. See the docstrings
    // under the main() function header for more info.
    //
    // Params:
    // :param k_vec: the normalized axis of rotation
    // :param B: the angle of rotation, in radians
    // :param v_vec: a vector to be rotated
    // :param v_rot: a vector to store the result of rotating v_vec an angle B about k_vec
    // :return: none

    int nk = k_vec.size();
    int nv = v_vec.size();
    assert(("vectors k and v must have the same length", nk==nv));
    
    // find (k ⨯ v) and (k·v)
    vector<float> kxv_vec(3);
    cross(k_vec, v_vec, kxv_vec);
    float kdv = std::inner_product(k_vec.begin(), k_vec.end(), v_vec.begin(), 0.0);
    
    // do rotation per-dimension
    float v;
    float k, k_x_v;
    for(int i=0; i<nk; ++i){
        v = v_vec[i];
        k = k_vec[i];
        k_x_v = kxv_vec[i];

        v_rot[i] = v*cos(B) + (k_x_v)*sin(B) + k*kdv*(1-cos(B));
    } 
}

//////////////////////////////////////////////////////
//
//             Reading functions
//
//////////////////////////////////////////////////////

int getLCSubdirs(string dir, vector<string> &subdirs) {
    // This function writes all of the subdirectory names present in a lightcone
    // output directory to the string vector subdirs. The assumptions are that 
    // each subdirectory name somewhere contains the character couple "lc", and
    // that no non-directory items lie under dir/. 
    //
    // Params:
    // :param dir: path to a lightcone output directory
    // :param subdirs: a vector to contain the subdirectory names under dir
    // :return: none
    
    // open dir
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening lightcone data files" << dir << endl;
        return errno;
    }

    // find all items within dir/
    while ((dirp = readdir(dp)) != NULL) {
        if (string(dirp->d_name).find("lc")!=string::npos){ 
            subdirs.push_back(string(dirp->d_name));
        }   
    }
    closedir(dp);
    return 0;
}


string getLCFile(string dir) {
    // This functions returns the header file present in a lightcone output 
    // step subdirectory (header files are those that are unhashed (#n)). 
    // This function enforces that only one file header is found, implying
    // that the output of only one single lightcone step is contained in the
    // directory dir/. In short, a step-wise directory structure is assumed, 
    // as described in the documentation comments under the function header 
    // for getLCSteps(). 
    // Assumptions are that the character couple "lc" appear somewhere in the 
    // file name, and that there are no subdirectories or otherwise unhashed
    // file names present in directory dir/. 
    //
    // Params:
    // :param dir: the path to the directory containing the output gio files
    // :return: the header file found in directory dir/ as a string

    // open dir/
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening lightcone data files" << dir << endl;
        return errno;
    }

    // find all header files in dir/
    vector<string> files;
    while ((dirp = readdir(dp)) != NULL) {
        if (string(dirp->d_name).find("lc") != string::npos & 
            string(dirp->d_name.find("#") == string::npos)){ 
            files.push_back(string(dirp->d_name));
        }   
    }
    assert(("Too many header files in this directory. LC Output files should be 
             separated by step-respective subdirectories", files.size() == 1)
    closedir(dp);
    return files[0];
}


int getLCSteps(int minStep, string dir, vector<string> &step_strings){
    // This function finds all simulation steps that are present in a lightcone
    // output directory that are above some specified minimum step number. The 
    // expected directory structure is that given in Figure 8 of Creating 
    // Lightcones in HACC; there should be a top directory (string dir) under 
    // which is a subdirectory for each step that ran through the lightcone code. 
    // The name of these subdirectories is expected to take the form:
    //
    // {N non-digit characters, somewhere containing "lc"}{N digits composing the step number}
    // 
    // For example, lc487, and lcGals487 are valid. lightcone487_output, is not.
    // The assumptions stated in the documentation comments under the 
    // getLCSubdirs() function header are of course made here as well.
    // 
    // Params:
    // :param minStep: the minimum step of interest (corresponding to the maximum
    //                 redshift desired to appear in the cutout)
    // :param dir: the path to a lightcone output directory. It is assumed that 
    //             the output data for each lightcone step are organized into 
    //             subdirectories. The expected directory structure is described 
    //             in the documentation comments under the function header for
    //             getLCSubdirs()
    // :param step_strings: a vector to contain steps found in the lightcone
    //                      output, as strings. The steps given are all of those 
    //                      that are >= minStep. However, if there is no step = 
    //                      minStep in the lc output, then accept the largest step 
    //                      that satisfies step < minStep. This ensures that users 
    //                      will preferntially recieve a slightly deeper cutout 
    //                      than desired, rather than slightly shallower, if the 
    //                      choice must be made. (e.g. if minStep=300, and the only 
    //                      nearby steps present in the output are 299 and 301, 
    //                      299 will be the minimum step written to step_strings)
    // :return: none

    // find all lc step subdirs
    vector<string> subdirs;
    getLCSubdirs(dir, subdirs);

    // extract step numbers from each subdirs
    vector<int> stepsAvail;
    for(int i=0; i<subdirs.size(); ++i){
        for(string::size_type j = 0; j < subdirs[i].size(); ++j){
            if( isdigit(subdirs[i][j]) ){
                stepsAvail.push_back( stoi(subdirs[i].substr(j)) );
            }
        }   
    }

    // identify the lowest step to push to step_strings (see remarks at 
    // function header)
    sort(stepsAvail.begin(), stepsAvail.end());
    for(int k=0; k<stepsAvail.size(); ++k){
        step_strings.push_back( to_string(stepsAvail[ stepsAvail.size() - (k+1) ]) );
        if( stepsAvail[ stepsAvail.size() - (k+1) ] <= minStep){
            break;
        }
    }
    return 0;
}

//////////////////////////////////////////////////////
//
//             Cutout function
//
//////////////////////////////////////////////////////

void processLC(string dir_name, string out_dir, vector<string> step_strings, 
               vector<float> theta_bounds, vector<float> phi_bounds){

    Buffers b;

    // find all lc sub directories for each step in step_strings
    cout << "Reading directory: " << dir_name << endl;
    vector<string> subdirs;
    getLCSubdirs(dir_name, subdirs);
    cout << "Found subdirs:" << endl;
    for (vector<string>::const_iterator i = subdirs.begin(); i != files.end(); ++i)
         cout << *i << ' ';
    cout << endl << endl;

    // find the prefix (chars before the step number) in the subdirectory names.
    // It is assumed that all subdirs have the same prefix.
    string subdirPrefix;
    for(string::size_type j = 0; j < subdirs[0].size(); ++j){
        if( isdigit(subdirs[0][j]) == false){
            subdirPrefix = subdirs[0].substr(0, j);
            break;
        }
    }

    // perform cutout on data from each lc output step
    size_t max_size = 0;
    int step;
    for (int i=0; i<step_strings.size();++i){
        
        // find header file
        cout<< "Working on step " << step_strings[i] << endl;
        step =atoi(step_strings[i].c_str());
        ostringstream file_name;
        file_name << dir_name << subdirPrefix << step_strings[i] << "/";
        file_name << getLCFile(dir_name << subdirPrefix << step_strings[i]);

        cout << "Opening file: " << file_name.str() << endl;
        GenericIO reader(MPI_COMM_SELF,file_name.str());
        reader.openAndReadHeader(GenericIO::MismatchRedistribute);
        
        // set size of buffers to be the size required by the largest data column 
        int nRanks = reader.readNRanks();
        size_t current_size;
        for (int j=0; j<nRanks; ++j) {
            current_size = reader.readNumElems(j);
            max_size = current_size > max_size ? current_size : max_size;
        }
        max_size +=10;
        cout<< "max size: " << max_size << endl; 
        b.x.resize(max_size);
        b.y.resize(max_size);
        b.z.resize(max_size);
        b.vx.resize(max_size);
        b.vy.resize(max_size);
        b.vz.resize(max_size);
        b.a.resize(max_size);
        b.id.resize(max_size);
        b.step.resize(max_size);
        b.rotation.resize(max_size);
        b.replication.resize(max_size);
        b.theta.resize(max_size);
        b.phi.resize(max_size);
        cout<<"done resizing"<<endl;
        
        ofstream id_file;
        ofstream theta_file;
        ofstream phi_file;
        ofstream a_file;
        ofstream x_file;
        ofstream y_file;
        ofstream z_file;
        ofstream vx_file;
        ofstream vy_file;
        ofstream vz_file;
        ofstream rotation_file;
        ofstream replication_file;

        ostringstream id_file_name;
        ostringstream theta_file_name;
        ostringstream phi_file_name;
        ostringstream a_file_name;
        ostringstream x_file_name;
        ostringstream y_file_name;
        ostringstream z_file_name;
        ostringstream vx_file_name;
        ostringstream vy_file_name;
        ostringstream vz_file_name;
        ostringstream rotation_file_name;
        ostringstream replication_file_name;

        id_file_name << out_dir << "/id." << step << ".bin";
        theta_file_name << out_dir << "/theta." << step << ".bin";
        phi_file_name << out_dir << "/phi." << step << ".bin";
        a_file_name << out_dir << "/a." << step << ".bin";
        x_file_name << out_dir << "/x."<< step <<".bin";
        y_file_name << out_dir << "/y."<< step <<".bin";
        z_file_name << out_dir << "/z."<< step <<".bin";
        vx_file_name << out_dir << "/vx."<< step <<".bin";
        vy_file_name << out_dir << "/vy."<< step <<".bin";
        vz_file_name << out_dir << "/vz."<< step <<".bin";
        vz_file_name << out_dir << "/rotation."<< step <<".bin";
        vz_file_name << out_dir << "/replication."<< step <<".bin";
        
        cout<<"starting to open files"<<endl;
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
        rotation_file.open(rotation_file_name.str().c_str(), ios::out | ios::binary);
        replication_file.open(replication_file_name.str().c_str(), ios::out | ios::binary);
        cout<<"done opening files"<<endl;
        
        reader.addVariable("x", &b.x[0]); 
        reader.addVariable("y", &b.y[0]); 
        reader.addVariable("z", &b.z[0]); 
        reader.addVariable("vx", &b.vx[0]); 
        reader.addVariable("vy", &b.vy[0]); 
        reader.addVariable("vz", &b.vz[0]); 
        reader.addVariable("a", &b.a[0]); 
        reader.addVariable("step", &b.step[0]); 
        reader.addVariable("id", &b.id[0]); 
        reader.addVariable("rotation", &b.rotation[0]); 
        reader.addVariable("replication", &b.replication[0]); 

        for (int j=0; j<nRanks; ++j) {
            size_t current_size = reader.readNumElems(j);
            cout << "Reading:" << current_size << endl;
            reader.readData(j);
    
            cout << "Converting positions..." << j+1 << "/" << nRanks << endl;
            for (int k=0; k<current_size; ++k) {

                if (b.x[k] > 0.0 && b.y[k] > 0.0 && b.z[k] > 0.0) {
                    float r = (float)sqrt(b.x[k]*b.x[k]+b.y[k]*b.y[k]+b.z[k]*b.z[k]);
                    b.theta[k] = acos(b.z[k]/r) * 180.0 / PI * ARCSEC;
                    b.phi[k]     = atan(b.y[k]/b.x[k]) * 180.0 / PI * ARCSEC;
                    if (b.theta[k] > theta_cut[0] && b.theta[k] < theta_cut[1] && 
                        b.phi[k] > phi_cut[0] && b.phi[k] < phi_cut[1] ) {
                        id_file.write( (char*)&b.id[k], sizeof(int64_t));
                        theta_file.write( (char*)&b.theta[k], sizeof(float));
                        phi_file.write( (char*)&b.phi[k], sizeof(float));
                        x_file.write((char*)&b.x[k],sizeof(float));
                        y_file.write((char*)&b.y[k],sizeof(float));
                        z_file.write((char*)&b.z[k],sizeof(float));
                        vx_file.write((char*)&b.vx[k],sizeof(float));
                        vy_file.write((char*)&b.vy[k],sizeof(float));
                        vz_file.write((char*)&b.vz[k],sizeof(float));
                        a_file.write( (char*)&b.a[k], sizeof(float));
                        rotation_file.write( (char*)&b.rotation[k], sizeof(int));
                        replication_file.write( (char*)&b.replication[k], sizeof(int32_t));
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
        a_file.close();
        rotation_file.close();
        replication_file.close();
    }
}

//////////////////////////////////////////////////////
//
//            driver function
//
//////////////////////////////////////////////////////

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
     
    // build step_strings vector by locating the step present in the lightcone
    // data directory that is nearest the redshift requested by the user
    float maxZ = argv[3];
    int minStep = zToStep(maxZ);    
    vector<string> step_strings;
    getLCSteps(minStep, input_lc_dir, step_strings)
    cout << "steps to include: ";

    // might note use all of these but whatever
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
    
    // there are two general use cases of this cutout code, as described in the 
    // docstring below the declaration of this main function. Here, exceptons are 
    // thrown to prevent confused input arguments which mix those two use cases.
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
    // Note to future self: strcmp() returns 0 if the input strings are equal. 
    for(int i=4; i<argc; ++i){

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
    cout << endl << endl;   
        
    // call overloaded processing function
    if(customHalo){
        processLC(input_lc_dir, out_dir, step_strings, haloPos, boxLength);
    }else{
        processLC(input_lc_dir, out_dir, step_strings, theta_cut, phi_cut);
    }

    MPI_Finalize();
    return 0;
}
