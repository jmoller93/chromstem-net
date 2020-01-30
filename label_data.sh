#!/bin/bash

# This is the script to generate the datasets
idir=`pwd`
cd data

# Validation, testing, and training fractions
val=0.15 # Same as testing
train=0.70

# Loop over different numbers of nucleosomes
for nnucl in {2..100..1}; do
    mkdir -p $idir/data/test/$nnucl
    mkdir -p $idir/data/train/$nnucl
    mkdir -p $idir/data/val/$nnucl
    cd $nnucl

    # Loop over 10 replicas
    for irep in {0..9..1}; do
        if [ -d $irep ]; then
            mkdir -p $idir/data/test/$nnucl/$irep
            mkdir -p $idir/data/train/$nnucl/$irep
            mkdir -p $idir/data/val/$nnucl/$irep
            rm $idir/data/test/$nnucl/$irep/*
            rm $idir/data/train/$nnucl/$irep/*
            rm $idir/data/val/$nnucl/$irep/*

            cd $irep

            # Remove all the labeled data
            if [[ -f label-1.csv ]]; then
                rm label*.csv
            fi

            # Remove all the data
            if [[ -f data-1.dat ]]; then
                rm data*.dat
            fi

            # Run the script to generate the data
            if [[ -f traj.dump ]]; then
                $idir/bin/calc_dna_voxel_density.exe traj.dump

                # Count the number of files generated
                shopt -s nullglob
                logfiles=(data*)
                count=${#logfiles[@]}
                maxval=`python -c "from math import floor; print(floor($val*$count))"`

                # Move files to training/validation/test data
                idx=0
                while [ $idx -le $maxval ]; do
                    frame=$((RANDOM%count))
                    # Concatenate all the data into one file
                    if [ -f data-${frame}.dat ]; then
                        # Add the working directory to the combined file
                        cmd="sed -i -e 's#^#data/val/${nnucl}/${irep}/#' label-${frame}.csv"
                        eval $cmd
                        cat label-${frame}.csv >> tmp && awk -F'#' 'NF!=2' tmp >> $idir/vals_label.csv && rm tmp
                        mv data-${frame}.dat $idir/data/val/$nnucl/$irep/
                        rm label-${frame}.csv
                        idx=$((idx+1))
                    fi
                done

                idx=0
                # Now the testing data
                while [ $idx -le $maxval ]; do
                    frame=$((RANDOM%count))
                    # Concatenate all the data into one file
                    if [ -f data-${frame}.dat ]; then
                        cmd="sed -i -e 's#^#data/test/${nnucl}/${irep}/#' label-${frame}.csv"
                        eval $cmd
                        cat label-${frame}.csv >> tmp && awk -F'#' 'NF!=2' tmp >> $idir/tests_label.csv && rm tmp
                        mv data-${frame}.dat $idir/data/test/$nnucl/$irep/
                        rm label-${frame}.csv
                        idx=$((idx+1))
                    fi
                done
            fi

            # Concatenate all the data into one file
            cat label*.csv >> tmp && awk -F'#' 'NF!=2' tmp > train_label.csv && rm tmp
            mv data* $idir/data/train/$nnucl/$irep/

            # Add the working directory to the combined file
            cmd="sed -i -e 's#^#data/train/${nnucl}/${irep}/#' train_label.csv"
            eval $cmd

            # Concatenate all the files into a master file
            if [[ -f train_label.csv  ]]; then
                cat train_label.csv >> ../train_label.csv
                rm train_label.csv
            fi

            # Remove the label files
            rm label*.csv
            cd ..
        fi
    done
    if [[ -f train_label.csv  ]]; then
        cat train_label.csv >> ../train_label.csv
        rm train_label.csv
    fi
    cd ..
done

mv train_label.csv ../trains_label.csv
cd $idir

cmd="sed -i -e 's/.$//' trains_label.csv"
eval $cmd
cmd="sed -i -e 's/.$//' vals_label.csv"
eval $cmd
cmd="sed -i -e 's/.$//' tests_label.csv"
eval $cmd

echo "Data now generated and labeled"

