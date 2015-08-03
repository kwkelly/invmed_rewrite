#!/bin/bash
MAX_QUEUE_DEFAULT=20
WAIT_TIME_DEFAULT=60
MAX_TIME_DEFAULT=01:00:00
DEPTHS_DEFAULT=4
M_DEFAULT=8
N_DEFAULT=8
Q_DEFAULT=6


usage="$(basename "$0") [-h] [-?] [-t -q -d -m -n] -- run job script

where:
-h  Show this help text
-?  Show this help text
-t  max run time (hh:mm:ss)
-q  Chebyshev polynomial order
-m  multipole order
-d  depth
-n  number of mpi tasks"

while getopts "h?t:q:d:m:n:" opt; do
	case "$opt" in
		h|\?)
		echo "$usage"
		exit 0
		;;
		t) MAX_TIME=$OPTARG
		;;
		d) DEPTHS=$OPTARG
		;;
		m) MS=$OPTARG
		;;
		q) QS=$OPTARG
		;;
		n) NS=$OPTARG
		;;
	esac
done
shift $((OPTIND-1))

MAX_TIME=${MAX_TIME:-$(echo $MAX_TIME_DEFAULT)}
DEPTHS=${DEPTHS:-$(echo $DEPTHS_DEFAULT)}
NS=${NS:-$(echo $N_DEFAULT)}
MS=${MS:-$(echo $M_DEFAULT)}
QS=${QS:-$(echo $Q_DEFAULT)}


for DEPTH in ${DEPTHS}
do
for Q in ${QS}
do
for M in ${MS}
do
for N in ${NS}
do
	while : ; do
		[[ $(squeue -u $USER | tail -n +1 | wc -l) -lt $MAX_QUEUE_DEFAULT ]] && break
		echo "Pausing until the queue empties enough to add a new one."
		sleep $WAIT_TIME_DEFAULT
	done
	JOBNAME=tests-$DEPTH-$Q-$M-$N
cat <<-EOS | sbatch
	#!/bin/bash

	#SBATCH -J $JOBNAME
	#SBATCH -o ../results/$JOBNAME.out
	#SBATCH -n $(( 2*$N ))
	#SBATCH -N $N
	#SBATCH -p development
	#SBATCH -t ${MAX_TIME:-$(echo $MAX_TIME_DEFAULT)}
	##SBATCH --mail-user=keith@ices.utexas.edu
	##SBATCH --mail-type=begin
	##SBATCH --mail-type=end
	#SBATCH -A PADAS
	cd ~/projects/invmed_rewrite/build/
	ibrun ./Gt_func_test -fmm_m $M -fmm_q $Q -min_depth $DEPTH -max_depth $DEPTH -dir /work/02370/kwkelly/maverick/files/results/

	exit 0
	EOS
done
done
done
done
