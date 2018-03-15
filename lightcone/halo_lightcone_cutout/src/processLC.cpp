#include "processLC.h"

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
//                Cutout function
//             	    Use Case 1
//           Custom theta - phi bounds
//
//////////////////////////////////////////////////////

void processLC(string dir_name, string out_dir, vector<string> step_strings, 
               vector<float> theta_cut, vector<float> phi_cut){

    Buffers b;

    // find all lc sub directories for each step in step_strings
    cout << "Reading directory: " << dir_name << endl;
    vector<string> subdirs;
    getLCSubdirs(dir_name, subdirs);
    cout << "Found subdirs:" << endl;
    for (vector<string>::const_iterator i = subdirs.begin(); i != subdirs.end(); ++i)
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
        string file_name;
        ostringstream file_name_stream;
        file_name_stream << dir_name << subdirPrefix << step_strings[i];
        file_name_stream << "/" << getLCFile(file_name_stream.str(), file_name);

        cout << "Opening file: " << file_name_stream.str() << endl;
        GenericIO reader(MPI_COMM_SELF, file_name_stream.str());
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
        a_file.open(a_file_name.str().c_str(), ios::out | ios::binary);
        x_file.open(x_file_name.str().c_str(), ios::out | ios::binary);
        y_file.open(y_file_name.str().c_str(), ios::out | ios::binary);
        z_file.open(z_file_name.str().c_str(), ios::out | ios::binary);
        vx_file.open(vx_file_name.str().c_str(), ios::out | ios::binary);
        vy_file.open(vy_file_name.str().c_str(), ios::out | ios::binary);
        vz_file.open(vz_file_name.str().c_str(), ios::out | ios::binary);
        rotation_file.open(rotation_file_name.str().c_str(), ios::out | ios::binary);
        replication_file.open(replication_file_name.str().c_str(), ios::out | ios::binary);
        cout<<"done opening files"<<endl;
        
        reader.addVariable("x", &b.x); 
        reader.addVariable("y", &b.y); 
        reader.addVariable("z", &b.z); 
        reader.addVariable("vx", &b.vx); 
        reader.addVariable("vy", &b.vy); 
        reader.addVariable("vz", &b.vz); 
        reader.addVariable("a", &b.a); 
        reader.addVariable("step", &b.step); 
        reader.addVariable("id", &b.id); 
        reader.addVariable("rotation", &b.rotation); 
        reader.addVariable("replication", &b.replication); 

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
//                Cutout function
//             	    Use Case 2
//               Custom halo cutout
//
//////////////////////////////////////////////////////

void processLC(string dir_name, string out_dir, vector<string> step_strings, 
               vector<float> halo_pos, float boxLength){

    Buffers b;

    // find all lc sub directories for each step in step_strings
    cout << "Reading directory: " << dir_name << endl;
    vector<string> subdirs;
    getLCSubdirs(dir_name, subdirs);
    cout << "Found subdirs:" << endl;
    for (vector<string>::const_iterator i = subdirs.begin(); i != subdirs.end(); ++i)
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
        string file_name;
        ostringstream file_name_stream;
        file_name_stream << dir_name << subdirPrefix << step_strings[i];
        file_name_stream << "/" << getLCFile(file_name_stream.str(), file_name);

        cout << "Opening file: " << file_name_stream.str() << endl;
        GenericIO reader(MPI_COMM_SELF, file_name_stream.str());
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
        a_file.open(a_file_name.str().c_str(), ios::out | ios::binary);
        x_file.open(x_file_name.str().c_str(), ios::out | ios::binary);
        y_file.open(y_file_name.str().c_str(), ios::out | ios::binary);
        z_file.open(z_file_name.str().c_str(), ios::out | ios::binary);
        vx_file.open(vx_file_name.str().c_str(), ios::out | ios::binary);
        vy_file.open(vy_file_name.str().c_str(), ios::out | ios::binary);
        vz_file.open(vz_file_name.str().c_str(), ios::out | ios::binary);
        rotation_file.open(rotation_file_name.str().c_str(), ios::out | ios::binary);
        replication_file.open(replication_file_name.str().c_str(), ios::out | ios::binary);
        cout<<"done opening files"<<endl;
        
        reader.addVariable("x", &b.x); 
        reader.addVariable("y", &b.y); 
        reader.addVariable("z", &b.z); 
        reader.addVariable("vx", &b.vx); 
        reader.addVariable("vy", &b.vy); 
        reader.addVariable("vz", &b.vz); 
        reader.addVariable("a", &b.a); 
        reader.addVariable("step", &b.step); 
        reader.addVariable("id", &b.id); 
        reader.addVariable("rotation", &b.rotation); 
        reader.addVariable("replication", &b.replication); 

        vector<float> theta_cut(2);
        vector<float> phi_cut(2);
        
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
