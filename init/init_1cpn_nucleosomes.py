#!/usr/bin/env python

# Subversion Info
#$Revision: 4478 $
#$LastChangedDate: 2017-04-06 10:56:10 -0500 (Thu, 06 Apr 2017) $

import sys
import math as m
import pdb
import numpy as np
import argparse
import sys
import os.path

from molecule import *
from vect_quat_util import *

# mape site names to numbers
typemap = {'nucl': 1,
           'bead': 2,
           'ghost': 3 }

def write_args(args,fnme):
   f=open(fnme,"w")
   command = ' '.join(sys.argv)
   f.write('# Inputs issued to command line\n')
   f.write("%s\n" %command)
   f.write('# All Parameters used\n')
   f.write("%s\n" %args)
   f.write("SVN $Revision: 4478 $")
   f.write("SVN $LastChangedDate: 2017-04-06 10:56:10 -0500 (Thu, 06 Apr 2017) $")
   f.close()


class Parameters(object):
    __slots__ = ('nucl_bp_unwrap',
                'charge_per_nucleosome',
                'bp_mass','charge_per_bp','twist_per_bp','rise_per_bp',
                'bead_mass','charge_per_bead','twist_per_bead','rise_per_bead',
                'nucl_mass',
                'ghost_mass',
                'basepair_per_bead',
                'nrl','nrlends',
                'lnucldna',
                'lengthscale',
                'dna_linker_length','dna_linker_length_ends',
                'dna_in_nucl',
                'bead_shape','nucl_shape','ghost_shape',
                'nucl_nucl_bond_offset',
                'nnucleosomes',
                'directory',)
    def __init__(self):
        self.dna_in_nucl = 147;
        self.directory = "."
        self.basepair_per_bead = 3 # 3 basepair per bead
        self.charge_per_nucleosome = 0;
        self.charge_per_bp = -1;
        self.twist_per_bp = 2.0*m.pi / 10.0 # radian
        self.rise_per_bp = 3.3 # Angstroms

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('-n','--nucleosomes',default=10,type=int, help='Number of nucleosomes')
  args = parser.parse_args()

  # structures to store data
  #geom = Geometry()
  param = Parameters()

  ##------------------------------------------------
  ## Input Values
  ##------------------------------------------------
  #geom.alpha = args.stemangle * m.pi / 180. # alpha that config is initialized to

  l = 1000.0 # Angstroms
  lhalf = 0.5*l

  param.nnucleosomes = args.nucleosomes
  density = param.nnucleosomes * np.power(l,-3.0)

  if density  > 1E-3:
      print("Warning! density %f > maximum density %f" % (density,0.001))

  n_per_side = int(np.ceil(np.power(param.nnucleosomes,1./3.)))
  if ( np.abs(np.power(param.nnucleosomes,1./3.)) - n_per_side) > 1e-3:
    print("Warning! specified number of nucleosomes is not a cube root of an integer")
    print(np.power(param.nnucleosomes,1./3.))
    print(n_per_side)
    pdb.set_trace()

  delta = l/n_per_side

  param.nucl_bp_unwrap = 0

  #------------------------------------------------
  # reference position and orientation
  #------------------------------------------------
  pos0 = np.array([-lhalf,-lhalf,-lhalf])
  quat0 = tu2rotquat(1e-5,[1,0,0])
  q = [0] * 4
  # fvu0 is the reference coordinate system for nucl and beads
  # to get current reference system, just rotate fvu0 by a quat
  fvu0 = np.eye(3)

  #------------------------------------------------
  # Setup molecule
  #------------------------------------------------
  molecule = Molecule()

  # manually specify some molecule parameters
  param.nucl_shape = [55,110,110]
  param.bp_mass = 650 #g/mol
  param.nucl_mass = 107616 + param.bp_mass*(147-param.nucl_bp_unwrap)   # Histone = 107616 g/mol, basepairs = 147

  # set masses
  molecule.atom_types.append(AtomType(1,param.nucl_mass))
  boxl = lhalf;
  molecule.set_box(-boxl, boxl, -boxl, boxl, -boxl,boxl)

  molecule.natom_type = 1;


  # ====================================
  # Main generation loop
  # ====================================

  # counts
  iellipsoid = 1

  # INFO
  # for each new site, assume pos, fvu and quat from the previous site are set

  for iz in range(n_per_side):
    for iy in range(n_per_side):
      for ix in range(n_per_side):
        if iellipsoid <= param.nnucleosomes:

          mytype = typemap['nucl']
          molid = iellipsoid + 1

          pos = pos0 + np.array([ix, iy,iz])*delta
          quat = np.random.uniform(0,1,4)
          quat = quat/np.sqrt(sum(quat**2)) #normalize
          molecule.ellipsoids.append(Ellipsoid(iellipsoid,mytype,pos,quat,param.charge_per_nucleosome,param.nucl_shape,molid))

          iellipsoid += 1


  # call funciton to set all bonded interactions
  #set_bonded_interactions(molecule)

  # function to position first nucleosome at origin
  #if (param.nnucleosomes != 0):
  #  align_with_1kx5(molecule)

  # josh for bonding all nucl together
  #if p_nucl_nucl_bond_offset:
  #  gen_nucl_nucl_bonds(molecule,param.nucl_nucl_bond_offset)

  if (not os.path.exists(param.directory)):
    os.mkdir(param.directory)

  #write_lammps_variables('in.variables',param,geom)
  molecule.write_dump("in.dump")
  #molecule.write_xyz("in.xyz")
  molecule.write_lammps_input("in.lammps")
  write_args(args,"%s/in.args" %   param.directory)
  molecule.write_psf("%s/in.psf" % param.directory)

if __name__ == "__main__":
    main()
