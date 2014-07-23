#!/bin/bash

#make ${EXEC} -j
if [ ! -f ${EXEC} ] ; then exit -1; fi;

export RESULT_DIR=${WORK_DIR}/results/${RESULT_SUBDIR}
mkdir -p ${RESULT_DIR}
#find ${RESULT_DIR} -type f -size 0 -exec rm {} \;

if command -v timeout >/dev/null; then
  export TIMEOUT="timeout";
else
  export TIMEOUT="scripts/.timeout3 -t ";
fi

eval      $nodes_;
eval      $cores_;
eval   $mpi_proc_;
eval    $threads_;
eval        $rho_;
eval    $ref_tol_;
eval  $min_depth_;
eval  $max_depth_;
eval      $fmm_q_;
eval      $fmm_m_;
eval  $gmres_tol_;
eval $gmres_iter_;
eval   $max_time_;

declare -a     args=();
declare -a    fname=();
for (( k=0; k<${#nodes[@]}; k++ )) ; do
  if [ "${nodes[k]}" == ":" ]; then continue; fi;
  args[$k]="";
  args[$k]="${args[k]} -rho ${rho[k]} -ref_tol ${ref_tol[k]} -min_depth ${min_depth[k]} -max_depth ${max_depth[k]}"
  args[$k]="${args[k]} -fmm_q ${fmm_q[k]} -fmm_m ${fmm_m[k]} -gmres_tol ${gmres_tol[k]} -gmres_iter ${gmres_iter[k]}"
  args[$k]="${args[k]} -ksp_gmres_modifiedgramschmidt -ksp_monitor";
  #args[$k]="${args[k]} -tree";
  #args[$k]="${args[k]} -vtk_order 0";

  case $HOSTNAME in
    *titan*) #titan.ccs.ornl.gov
        fname[$k]="host_titan";
      ;;
    *stampede*) #stampede.tacc.utexas.edu
        fname[$k]="host_stampede";
      ;;
    *ls4*) #lonestar.tacc.utexas.edu
        fname[$k]="host_lonestar";
      ;;
    *ronaldo*) #ronaldo.ices.utexas.edu
        fname[$k]="host_ronaldo";
      ;;
    *) # none of the known machines
        fname[$k]="host_${HOSTNAME}";
  esac
  fname[$k]="${fname[k]}_n${nodes[k]}_mpi${mpi_proc[k]}_omp${threads[k]}";
  fname[$k]="${fname[k]}_test_case${test_case[k]}_pt_cnt${pt_cnt[k]}_pt_rad${pt_rad[k]}_jwid${jump_width[k]}_rho${rho[k]}_ref_tol${ref_tol[k]}_min_depth${min_depth[k]}_max_depth${max_depth[k]}"
  fname[$k]="${fname[k]}_fmm_q${fmm_q[k]}_fmm_m${fmm_m[k]}_gmres_tol${gmres_tol[k]}_gmres_iter${gmres_iter[k]}"
done
export     args_="$(declare -p     args)";
export    fname_="$(declare -p    fname)";

for (( k=0; k<${#nodes[@]}; k++ )) ; do
  if [ "${nodes[k]}" == ":" ] ||
     [ -f ${RESULT_DIR}/$(basename ${EXEC})_${fname[k]}.out ]; then
    #grep "outSize" ${RESULT_DIR}/$(basename ${EXEC})_${fname[k]}.out;
    continue;
  fi;
  for (( j=0; j<$k; j++ )) ; do
    if [ "${nodes[k]}" == "${nodes[j]}" ] &&
       [ "${mpi_proc[k]}" == "${mpi_proc[j]}" ] &&
       [ ! -f ${RESULT_DIR}/$(basename ${EXEC})_${fname[j]}.out ]; then
      continue 2;
    fi
  done;
  TOTAL_TIME=0;
  for (( j=0; j<${#nodes[@]}; j++ )) ; do
    if [ "${nodes[k]}" == "${nodes[j]}" ] &&
       [ "${mpi_proc[k]}" == "${mpi_proc[j]}" ] &&
       [ ! -f ${RESULT_DIR}/$(basename ${EXEC})_${fname[j]}.out ]; then
      TOTAL_TIME=$(( ${TOTAL_TIME} + ${max_time[j]} ))
    fi
  done;

  export    NODES=${nodes[k]};    # Number of compute nodes.
  export    CORES=${cores[k]};    # Number of cores per node.
  export MPI_PROC=${mpi_proc[k]}; # Number of MPI processes.
  export  THREADS=${threads[k]};  # Number of threads per MPI process.
  export    FNAME=${RESULT_DIR}/$(basename ${EXEC})_nds${NODES}_mpi${MPI_PROC}

  #Submit Job
  if [[ ! -z "$PBS_NODEFILE" ]] || [[ ! -z "$SLURM_NODELIST" ]];  then
    ./scripts/.job.sh
  else
    case $HOSTNAME in
      *titan*) #titan.ccs.ornl.gov (Portable Batch System)
          qsub -l nodes=${NODES} \
               -o ${FNAME}.out -e ${FNAME}.err \
               -l walltime=${TOTAL_TIME} \
               ./scripts/.job.titan
        ;;
      *stampede*) #stampede.tacc.utexas.edu (Slurm Batch)
          if (( ${TOTAL_TIME} > 14400 )); then TOTAL_TIME="14400"; fi
          #if (( ${NODES} > 128 )) ; then continue; fi;
          sbatch -N${NODES} -n${MPI_PROC} \
                 -o ${FNAME}.out -e ${FNAME}.err -D ${PWD} \
                 --time=00:00:${TOTAL_TIME} \
                 ./scripts/.job.stampede
        ;;
      *ls4*) #lonestar.tacc.utexas.edu (Sun Grid Engine)
          qsub -pe $((${MPI_PROC}/${NODES}))way $((${NODES}*${CORES})) \
               -o ${FNAME}.out -e ${FNAME}.err \
               -l h_rt=${TOTAL_TIME} \
               ./scripts/.job.lonestar
        ;;
      *ronaldo*) #ronaldo.ices.utexas.edu (Portable Batch System)
          qsub -l nodes=${NODES}:ppn=$((${MPI_PROC}/${NODES})) \
               -o ${FNAME}.out -e ${FNAME}.err \
               -l walltime=${TOTAL_TIME} \
               ./scripts/.job.ronaldo
        ;;
      *) # none of the known machines
        if command -v qsub >/dev/null; then # Portable Batch System
          qsub -l nodes=${NODES}:ppn=$((${MPI_PROC}/${NODES})) \
               -o ${FNAME}.out -e ${FNAME}.err \
               -l walltime=${TOTAL_TIME} \
               ./scripts/.job.qsub
        elif command -v sbatch >/dev/null; then # Slurm Batch
          sbatch -N${NODES} -n${MPI_PROC} \
                 -o ${FNAME}.out -e ${FNAME}.err -D ${PWD} \
                 --time=${TOTAL_TIME} \
                 ./scripts/.job.sbatch
        else # Shell
          ./scripts/.job.sh
        fi
    esac
  fi

  #Exit on error.
  if (( $? != 0 )) ; then continue; fi;
  for (( j=0; j<${#nodes[@]}; j++ )) ; do
    if [ "${nodes[k]}" == "${nodes[j]}" ] &&
       [ "${mpi_proc[k]}" == "${mpi_proc[j]}" ] &&
       [ ! -f ${RESULT_DIR}/$(basename ${EXEC})_${fname[j]}.out ]; then
      touch ${RESULT_DIR}/$(basename ${EXEC})_${fname[j]}.out;
    fi
  done;
done;

