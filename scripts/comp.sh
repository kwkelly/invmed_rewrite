#!/bin/bash
#SBATCH -J build
#SBATCH -o build.o%j 
#SBATCH -n 2
#SBATCH -p normal
#SBATCH -t 00:30:00
#SBATCH --mail-user=keith@ices.utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A PADAS
cd ~/projects/invmed_rewrite/build/
cmake ..
make
