#!/bin/bash
#SBATCH -J test
#SBATCH -o test.o%j 
#SBATCH -n 4
#SBATCH -N 4
#SBATCH -p normal
#SBATCH -t 00:10:00
##SBATCH --mail-user=keith@ices.utexas.edu
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
#SBATCH -A PADAS
cd ~/projects/invmed_rewrite/build/
ibrun ./test -fmm_m 8 -fmm_q 10 -min_depth 1 -max_depth 3
#ibrun ./dist_test
