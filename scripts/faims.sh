#!/bin/bash
#SBATCH -J faims
#SBATCH -o faims.o%j 
#SBATCH -n 1
#SBATCH -p normal
#SBATCH -t 02:00:00
#SBATCH --mail-user=keith@ices.utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A PADAS
cd ~/projects/invmed_rewrite/build/
./faims -fmm_m 8 -fmm_q 10 -min_depth 6 -max_depth 6 -k 35 -R_d 50 -N_pts 15
