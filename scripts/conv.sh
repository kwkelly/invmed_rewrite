#!/bin/bash

for ALPHA in `seq 7 7`
do
cat <<EOS | qsub -
#!/bin/bash

# declare a name for this job to be sample_job
#PBS -N conv
# request the queue (enter the possible names, if omitted, serial is the default)
#PBS -q nb
#PBS -V
#PBS -S /bin/bash
#PBS -l nodes=1
#PBS -l walltime=45:00:00
#PBS -m bea
#PBS -M keith@ices.utexas.edu
#PBS -o falpha$ALPHA.o
#PBS -e falpha$ALPHA.e
set -x # echo on

#cd $PBS_O_WORKDIR
# run the program
cd Documents/PADAS/Projects/pvfmm/invmed3/
./bin/invmed -fmm_m 8 -fmm_q 10 -ref_tol 1e-6 -max_depth 6 -tol 1e-6 -obs 1 -alpha 1e-$ALPHA -iter 5000 -ksp_monitor_true_residual -min_depth 2 \
&& mkdir results/resultsfa$ALPHA \
&& mv results/sol.pvtu results/resultsfa$ALPHA\
&& mv results/sol000000.vtu results/resultsfa$ALPHA
exit 0

EOS
done
