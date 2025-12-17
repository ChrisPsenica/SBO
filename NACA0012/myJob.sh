#!/bin/bash

#SBATCH --time=00:20:00	     	            # walltime limit (HH:MM:SS)
#SBATCH --nodes=1		                    # number of nodes
#SBATCH --ntasks-per-node=16 	            # 32 processor core(s) per node 
#SBATCH --job-name="EGO"                    # job name
#SBATCH --output="log-%j.txt"	            # job standard output file (%j replaced by job id)
#SBATCH --constraint=intel                  # use intel cores to avoid bad termination on HPC
#SBATCH --mail-user=cpsenica@iastate.edu    # who to email when the job is done
#SBATCH --mail-type=ALL                     # mail type
#SBATCH --constraint=nova22                 # avoid speedy and novaWide

#------------------------------------------------------------------------------

# load framework
. /work/phe/DAFoam_Nova_Gcc/latest/loadDAFoam.sh

# run simulation
python EGO.py

#------------------------------------------------------------------------------
