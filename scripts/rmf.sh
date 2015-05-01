#!/bin/bash
MAX_QUEUE=20
WAIT_TIME=1

for DEPTH in `seq 1 25`
do
	for R_D in 25
	do
		for R_S in 25
		do
			for N_PTS in 5
			do
				while : ; do
					[[ $(squeue -u $USER | tail -n +1 | wc -l) -lt $MAX_QUEUE ]] && break
					echo "Pausing until the queue empties enough to add a new one."
					sleep $WAIT_TIME
				done
				R_DN=$(($R_D<$N_PTS*$N_PTS?$R_D:$N_PTS*$N_PTS))
				R_SN=$(($R_S<$N_PTS*$N_PTS?$R_S:$N_PTS*$N_PTS))
				echo $R_DN
cat <<-EOS | sbatch
				#!/bin/bash

				#SBATCH -J conv-$DEPTH-$R_D-$N_PTS
				#SBATCH -o conv-$DEPTH-$R_D-$N_PTS.6.out
				#SBATCH -n 16
				#SBATCH -N 16
				#SBATCH -p gpu
				#SBATCH -t 00:60:00
				##SBATCH --mail-user=keith@ices.utexas.edu
				##SBATCH --mail-type=begin
				##SBATCH --mail-type=end
				#SBATCH -A PADAS
				cd ~/projects/invmed_rewrite/build/
				ibrun ls

				exit 0
				EOS
			done
		done
	done
done
