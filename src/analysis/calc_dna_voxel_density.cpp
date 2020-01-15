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

    // Initialize the histogram
    Hist hist;
    hist.set_nbins(34);
    hist.set_range(-500.0,500.0);

    //Loop through the dump file using the parser
    for(size_t i=0; i<timestep; i++) {
        //Initialize dna coordinates
        std::vector<std::vector<double>> dna_coords;
        std::vector<std::vector<double>> nucl_orient;

        //Make sure to move to the next frame
        parser.next_frame();

        //Get the dna coordinates
        dna_coords  = parser.get_dna_coord();
        nucl_orient = parser.get_vect('f');
 
        // Create the output label file
        std::string fnme = "label-";
        fnme += std::to_string(i);
        fnme += ".csv";
        std::ofstream file(fnme, std::ofstream::app);
        file << "# Nucl Coordinate file <x> <y> <z> <f1> <f2> <f3>"<<std::endl;
        file << "data-"<<i<<".dat,";
        for (size_t j=0; j<parser.coords_.size(); j++) {
            for (size_t k=0; k<3; k++) {file<<parser.coords_[j][k]<<",";}
            //for (size_t k=0; k<3; k++) {file<<nucl_orient[j][k]<<", "<<std::setw(3);} Remove quaternions for now
        }
        file<<std::endl;
        file.close();

        // Create the output data file
        for (size_t j=0; j<dna_coords.size(); j++) 
            hist.update_hist(dna_coords[j]);
        hist.print_hist("data-",i);
        hist.clear_hist();
    }  
}
