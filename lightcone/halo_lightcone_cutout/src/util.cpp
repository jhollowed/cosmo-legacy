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
