#include "hist.h"
#include "trajectory_iterator.h"

#define TWID 500

int main(int argc, char**argv){
    
    if (argc != 2) {
        std::cout<<"Usage: "<<argv[0]<<" <dump file>"<<std::endl;
        exit(1);
    }

    int timestep, natom;

    //Set up all vectors needed for the trajectory parser class
    TrajectoryIterator parser;
    std::vector<int> atom_types;
    std::vector<float> box_dim;
       
    //Load the trajectory into the parser
    parser.load_dump(argv[1]); 
    //Get the number of frames
    timestep = parser.get_numFrames();
    std::cout<<timestep<<std::endl;
    //Get the vector for the types of atoms
    atom_types = parser.get_types(); 
    //Get the number of atoms
    natom = parser.get_numAtoms();
    box_dim = parser.get_boxDim();

    //A few extraneous variables to find the nucleosomes
    int nnucl; 
    std::vector<int> nucl_ids;
    bool firstframe = true;

    //Loop through the dump file using the parser
    for(size_t i=0; i<timestep; i++) {

        // Get the nucleosome indexes
        if (firstframe){
            nnucl = 0;
            std::vector<int> types = parser.get_types();
            for (size_t j=0;j<natom;j++){
              if (types[j] == 1){ //is nucleosome
                nucl_ids.push_back(j+1);
                nnucl++;
              }
            }
        }

        //Initialize dna coordinates
        std::vector<std::vector<double>> x1_vects;
        std::vector<std::vector<double>> x2_vects;

        //Make sure to move to the next frame
        parser.next_frame();

        //Get the dna coordinates
        x1_vects = parser.get_vect('b');
        x2_vects = parser.get_vect('d');
 
        // Create the output label file
        std::string fnme = "bbox-";
        fnme += std::to_string(i);
        fnme += ".csv";
        std::ofstream file(fnme, std::ofstream::app);
        file << "# Bounding boxes file <x1> <x2> <y1> <y2> <z1> <z2>"<<std::endl;
        for (size_t j=0; j<nucl_ids.size(); j++) {
            file << "Nucleosome, ";
            for (size_t k=0; k<3; k++) {
                file<<x1_vects[j][k]*87.5+parser.coords_[j][k]<<", ";
                file<<x2_vects[j][k]*87.5+parser.coords_[j][k]<<", ";
            }
            file<<std::endl;
            //for (size_t k=0; k<3; k++) {file<<nucl_orient[j][k]<<", "<<std::setw(3);} Remove quaternions for now
        }
        file<<std::endl;
        file.close();

        firstframe = false;
    }  
}
