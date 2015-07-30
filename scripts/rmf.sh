#!/bin/bash
MAX_QUEUE=20
WAIT_TIME=60
MAX_TIME=01:00:00

for DEPTH in 4
do
	for R_D in 10
	do
		for R_S in 10
		do
			for N_S in 25
			do
				#for N_D in 4
				#do
				for K in 10
				do
					while : ; do
						[[ $(squeue -u $USER | tail -n +1 | wc -l) -lt $MAX_QUEUE ]] && break
						echo "Pausing until the queue empties enough to add a new one."
						sleep $WAIT_TIME
					done
					MIN_SD=$(($R_D<$R_S?$R_S:$R_S))
					JOBNAME=faims-$DEPTH-$N_S-$R_D-$R_S-$K
					#JOBNAME=compute_kernel
					if [ "$MIN_SD" -ge  "$K" ]; then
cat <<-EOS | sbatch
					#!/bin/bash

					#SBATCH -J $JOBNAME
					#SBATCH -o $JOBNAME.out
					#SBATCH -n 4
					#SBATCH -N 4
					#SBATCH -p gpu
					#SBATCH -t $MAX_TIME
					##SBATCH --mail-user=keith@ices.utexas.edu
					##SBATCH --mail-type=begin
					##SBATCH --mail-type=end
					#SBATCH -A PADAS
					cd ~/projects/invmed_rewrite/build/
					ibrun ./faims -fmm_m 8 -fmm_q 6 -min_depth $DEPTH -max_depth $DEPTH -N_s $N_S -N_d $N_S -R_d $R_D -R_s $R_S -R_b $R_S -k $K

					exit 0
					EOS
				fi
				done
				done
			#done
		done
	done
done
