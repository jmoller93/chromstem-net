#!/bin/bash

# This is the script to generate the datasets
idir=`pwd`
mkdir -p wdir && cd wdir

# Loop over different numbers of nucleosomes
for nnucl in {2..100..1}; do
    mkdir -p $nnucl
    cd $nnucl

    # Loop over 10 replicas
    for irep in {0..9..1}; do
        mkdir -p $irep
        cd $irep

        # If file doesn't exist, make new data
        if [ ! -f traj.dump ]; then
            # Copy seed files here
            cp $idir/seed/* .

            # Change the random number initializer
            cmd="sed -i -e 's/RANDOM/$RANDOM/g' in.nucleosomes"
            eval $cmd

            # Run the initialization script
            python2 $idir/init/init_1cpn_nucleosomes.py -n $nnucl

            # Run lammps
            lmp_mpi < in.nucleosomes
        fi
        cd ..
    done

    cd ..
done

cd $idir

