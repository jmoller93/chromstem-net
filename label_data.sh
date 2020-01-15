#!/bin/bash

# This is the script to generate the datasets
idir=`pwd`
cd data

# Loop over different numbers of nucleosomes
for nnucl in {5..65..5}; do
    cd $nnucl

    # Loop over 10 replicas
    for irep in {0..9..1}; do
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
        fi

		# Concatenate all the data into one file
		cat label*.csv >> merged_label.csv && awk -F'#' 'NF!=2' merged_label.csv > output_label.csv

        # Add the working directory to the combined file
        cmd="sed -i -e 's#^#data/${nnucl}/${irep}/#' output_label.csv"
        eval $cmd
        cmd="sed -i -e 's/.$//' output_label.csv"
        eval $cmd

        # Concatenate all the files into a master file
        if [[ -f output_label.csv  ]]; then
            cat output_label.csv >> ../output_label.csv
            rm *label*.csv
        fi

        cd ..
    done
    if [[ -f output_label.csv  ]]; then
        cat output_label.csv >> ../output_label.csv
        rm output_label.csv
    fi
    cd ..
done

cd $idir

