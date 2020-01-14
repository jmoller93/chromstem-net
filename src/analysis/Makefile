CC =     g++
BINDIR = '../../bin'
LIBRY      =  -L/usr/local/lib -lm 
INCLD      = -I../include 

DEBUG ?= 0
ifeq ($(DEBUG),1)
	CFLGS =  -ggdb -Wall -std=c++11
else
	CFLGS =  -O2 -finline-functions -funroll-loops -std=c++11
endif

all: calc_dna_voxel_density

calc_dna_voxel_density : calc_dna_voxel_density.cpp
	${CC} ${CFLGS} ${INCLD} calc_dna_voxel_density.cpp -o  ${BINDIR}/calc_dna_voxel_density.exe

