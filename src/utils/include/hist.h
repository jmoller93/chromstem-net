#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <iostream>

class Hist {
    private:
        // Set variables within the class
        int nbins_ = 0; 
        float binSize_;
        float minRange_ = 0.0;
        float maxRange_ = 0.0;
        float l_ = 0.0;
        std::vector<size_t> x_;
        
        // Private utility functions
        std::vector<size_t> coord_to_bin(const std::vector<double>&);
        std::vector<float>  bin_to_coord(size_t);
        std::vector<int>    bin_to_index(size_t idx);
    public:
        void set_nbins(int);
        void set_range(float,float);
        void update_hist(const std::vector<double>&);
        void print_hist(std::string, int);
        void clear_hist(void);
};

// Set the number of bins
void Hist::set_nbins(int nbins) {
    nbins_ = nbins; 
    x_.resize(nbins_*nbins_*nbins_);
    clear_hist();
    if (l_) {binSize_ = (float) (l_)/(nbins_);}
    return;
};

// Set the number of bins
void Hist::set_range(float min, float max) {
    minRange_ = min;
    maxRange_ = max;
    l_ = maxRange_ - minRange_;
    if (nbins_) {binSize_ = (float) (l_)/(nbins_);}
    return;
};

// Convert Cartesian to bin coordinates
std::vector<size_t> Hist::coord_to_bin(const std::vector<double>& r) {
    std::vector<size_t> idx(3,0);
    for(size_t i=0; i<r.size(); i++) {
        float x = 0;
        if (r[i] < -l_*0.5) {x = r[i]+l_;} 
        else if (r[i] >= l_*0.5) {x = r[i]-l_;} 
        else {x = r[i];} 
        idx[i] = (x-minRange_)*(float)((nbins_-1)/l_);
    }
    return idx;
};

// Convert bin to Cartesian
std::vector<float> Hist::bin_to_coord(size_t idx) {
    std::vector<float> r(3,0);
    int kdx = idx;
    for(int i=2; i<r.size(); i--) {
        int jdx = kdx / pow(nbins_,i);
        r[i] = (float) jdx * l_ / (nbins_-1) + minRange_;
        kdx  = kdx % (int) pow(nbins_,i);
    }
    return r;
};

// Convert bin to Cartesian
std::vector<int> Hist::bin_to_index(size_t idx) {
    std::vector<int> r(3,0);
    int kdx = idx;
    for(int i=2; i<r.size(); i--) {
        r[i] = kdx / pow(nbins_,i);
        kdx  = kdx % (int) pow(nbins_,i);
    }
    return r;
};

// Update the histogram
void Hist::update_hist(const std::vector<double>& r) {
    std::vector<size_t> idx;
    idx = coord_to_bin(r);
    x_[idx[2]*nbins_*nbins_+idx[1]*nbins_+idx[0]] += 3;
    return;
};

// Print the histogram to data file
void Hist::print_hist(std::string fnme, int tstep) {
    fnme += std::to_string(tstep) + ".dat";
    std::ofstream file(fnme,std::ofstream::app);
    file<<"# DATA FOR TIMESTEP "<<tstep<<std::endl;
    file<<nbins_<<std::setw(10)<<nbins_<<std::setw(10)<<nbins_<<std::setw(10)<<"0"<<std::endl;

    std::vector<int> r; 
    for(size_t i=0; i<x_.size(); i++) {
        if (x_[i] != 0) {
            r = bin_to_index(i);
            file<<r[0]<<std::setw(10)<<r[1]<<std::setw(10)<<r[2]<<std::setw(10)<<(float) x_[i]/pow(0.1*binSize_,3.0)<<std::endl;
        }
    }
    file.close();
    return;
};

// Clear the histogram
void Hist::clear_hist(void) {
    std::fill(x_.begin(),x_.end(),0);
    return;
};
