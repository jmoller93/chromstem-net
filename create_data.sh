#!/bin/bash

# This is the script to generate the datasets
idir=`pwd`
mkdir wdir && cd wdir

# Loop over different numbers of nucleosomes
for nnucl in {5..100..5}; do
    mkdir $nnucl && cd $nnucl

    # Loop over 10 replicas
    for irep in {0..9..1}; do
        mkdir $irep && cd $irep

        # Copy seed files here
        cp $idir/seed/* .

        # Change the random number initializer
        cmd="sed -i -e 's/RANDOM/$RANDOM/g' in.liquidcrystal"
        eval $cmd

        # Run the initialization script
        python2 $idir/init/init_1cpn_nucleosomes.py -n $nnucl

        # Run lammps
        lmp_mpi < in.nucleosomes

        cd ..
    done

    cd ..
done

cd $idir

