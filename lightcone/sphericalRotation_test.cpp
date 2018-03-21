#include <vector>
#include <numeric>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <dirent.h>
#include <algorithm>
#include <locale>
#include <errno.h>
#include <string>

using namespace std;

vector<float> cross(const vector<float> & v1,
                    const vector<float> & v2,
                    vector<float> & v1xv2){
    // This function returns the cross product of the three dimensional
    // vectors v1 and v2, and writes the result at the location referenced 
    // by arg v1xv2.

    int n1 = v1.size();
    int n2 = v2.size();
    assert(("vectors v1 and v2 must have the same length", n1 == n2));

    v1xv2[0] = v1[1]*v2[2] - v1[2]*v2[1];
    v1xv2[1] = -(v1[0]*v2[2] - v1[2]*v2[0]);
    v1xv2[2] = v1[0]*v2[1] - v1[1]*v2[0];
    return v1xv2;

}

void normCross(const vector<float> & a,
               const vector<float> & b,
               vector<float> & k){
    // This function returns the normalized cross product of the three-dimensional
    // vectors a and b. The result is written at the location referenced by arg k
    // The notation is chosen to match that of the Rodrigues rotation formula for 
    // the rotation vector k, rather than matching the notation of cross() above. 
    // Feel free to contact me with urgency if you find this issue troublesome.

    int na = a.size();
    int nb = b.size();
    assert(("vectors a and b must have the same length", na==nb));

    vector<float> axb(3);
    cross(a, b, axb);
    float mag_axb = sqrt(std::inner_product(axb.begin(), axb.end(), axb.begin(), 0.0));
    for(int i=0; i<na; ++i){ k[i] = axb[i] / mag_axb; }
}

void rotate(const vector<float> & k_vec,
            const float B,
            const vector<float> & v_vec,
            vector<float> & v_rot){
    // This function implements the Rodrigues rotation formula. k is the axis
    // of rotation, whch can be returned from normCross(), B is the angle of 
    // rotation, and v_vec is a cartesian position vector to be rotated an 
    // angle B about k. The rotated vector is written at the location referenced
    // by arg v_rot. See the docstrings under main() for more info.

    int nk = k_vec.size();
    int nv = v_vec.size();
    assert(("vectors k and v must have the same length", nk==nv));


    // find (k ⨯ v) and (k·v)
    vector<float> kxv_vec(3);
    cross(k_vec, v_vec, kxv_vec);
    float kdv = std::inner_product(k_vec.begin(), k_vec.end(), v_vec.begin(), 0.0);

    // do rotation
    float v, k, k_x_v;
    for(int i=0; i<nk; ++i){
        v = v_vec[i];
        k = k_vec[i];
        k_x_v = kxv_vec[i];

        v_rot[i] = v*cos(B) + (k_x_v)*sin(B) + k*kdv*(1-cos(B));
    }
}

float zToStep(float z, int totSteps=499, float maxZ=200.0){
    // Function to convert a redshift to a step number
    // Note-- the initial conditions are not a step! totSteps 
    // should be the maximum snapshot number

    float amin = 1/(maxZ + 1);
    float amax = 1.0;
    float adiff = (amax-amin)/(totSteps-1);
    
	float a = 1/(1+z);
	cout << adiff << endl;
    int step = floor((a-amin) / adiff);
	return step;
}

int getLCSteps(int minStep, vector<string> & step_strings, string dir){
    // This function writes lightcone steps that are present in the lightcone
    // output directory, input_lc_dir (argv[1]) to the location given as 
    // step_strings. 
	// The steps given are all of those that are >= minStep. If 
    // there is no step = minStep in the lc output, then accept the largest
    // step that satisfies step < minStep. This ensures that users will 
    // preferntially recieve a slightly deeper cutout than desired, rather than 
    // slightly shallower, if the choice must be made.

    // open lightcone output dir
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening lightcone data files" << dir << endl;
        return errno;
    }

    // find all lc step subdirs
    vector<string> subdirs;
    while ((dirp = readdir(dp)) != NULL) {
        if (string(dirp->d_name).find("lc") != string::npos){           
            subdirs.push_back(string(dirp->d_name));
        }
    }
    closedir(dp);

    // extract step numbers from each subdirs
	vector<int> stepsAvail;
    for(int i=0; i<subdirs.size(); ++i){
        for(string::size_type j = 0; j < subdirs[j].size(); ++j){
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

void test(int a, int b = 4){
    cout << a + b << endl;
}

int main(){

    vector<float> a(3);
    a[0] = 0.49960183664463337; 
	a[1] = 0.49999984146591736; 
	a[2] = 0.7073882691671998;
    float ada = std::inner_product(a.begin(), a.end(), a.begin(), 0.0);
    float mag_a = sqrt(ada);
    cout << "a . a: " << mag_a << endl; 
    vector<float> b(3);
    b[0] = mag_a, 
	b[1] = 0;
	b[2] = 0;

	float mag_b = sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
	float adb = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
	float B = acos(adb / (mag_a*mag_b));

	vector<float> k(3);
    normCross(a, b, k);

	vector <float> v(3);
	v[0] = a[0]; 
	v[1] = a[1]; 
	v[2] = a[2];
	vector<float> v_rot(3);
	rotate(k, B, v, v_rot);
	cout << v_rot[0] << ", " << v_rot[1] << ", " << v_rot[2] << endl;

	float z = 3.0;
	int s = zToStep(z);
	cout << "z to step: " << z << " --> " << s << endl;

    vector<string> step_strings;
	string lc_output_dir = "/home/joe/lightcone/";
	int minStep = 498;
	getLCSteps(minStep, step_strings, lc_output_dir);
	for(int i = 0; i < step_strings.size(); ++i){cout << step_strings[i] << endl;}
    test(3);
	return 0;
}
