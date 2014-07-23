#!/bin/bash

eval    $nodes_;
eval    $cores_;
eval $mpi_proc_;
eval  $threads_;
eval $max_time_;
eval    $fname_;
eval     $args_;

WORK_DIR=${PWD}
cd ${WORK_DIR}

#export KMP_AFFINITY=verbose
for (( k=0; k<${#nodes[@]}; k++ )) ; do
  if [ "${nodes[k]}" == "${NODES}" ] &&
     [ "${mpi_proc[k]}" == "${MPI_PROC}" ]; then
    export OMP_NUM_THREADS=${threads[k]};

    # MIC async
    EXEC_=${EXEC};
    FNAME=${RESULT_DIR}/$(basename ${EXEC_})_${fname[k]}.out;
    if [ -f ${EXEC_} ] && [ ! -s ${FNAME} ] ; then
      printf '%*s\n\n' "100" ' ' | tr ' ' "#" | tee -a ${FNAME};
      printf "COMMAND: ${EXEC_} ${args[k]}\n" | tee -a ${FNAME};

      if [[ -n "$PBS_NODEFILE" ]] ;  then 
        ${TIMEOUT} ${max_time[k]} time mpirun -np ${mpi_proc[k]} \
          --hostfile ${PBS_NODEFILE} --bysocket --bind-to-socket  ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      elif [[ -n "$SLURM_NODELIST" ]] ;  then
        ${TIMEOUT} ${max_time[k]} time ibrun tacc_affinity ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      else
        ${TIMEOUT} ${max_time[k]} time mpirun -np ${mpi_proc[k]}  ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      fi
      printf '\n%*s\n\n' "100" ' ' | tr ' ' "#" | tee -a ${FNAME};
    fi;

    # GPU async
    EXEC_=${EXEC}_gpu;
    FNAME=${RESULT_DIR}/$(basename ${EXEC_})_${fname[k]}.out;
    if [ -f ${EXEC_} ] && [ ! -s ${FNAME} ] ; then
      printf '%*s\n\n' "100" ' ' | tr ' ' "#" | tee -a ${FNAME};
      printf "COMMAND: ${EXEC_} ${args[k]}\n" | tee -a ${FNAME};

      if [[ -n "$PBS_NODEFILE" ]] ;  then 
        ${TIMEOUT} ${max_time[k]} time mpirun -np ${mpi_proc[k]} \
          --hostfile ${PBS_NODEFILE} --bysocket --bind-to-socket  ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      elif [[ -n "$SLURM_NODELIST" ]] ;  then
        ${TIMEOUT} ${max_time[k]} time ibrun tacc_affinity ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      else
        ${TIMEOUT} ${max_time[k]} time mpirun -np ${mpi_proc[k]}  ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      fi
      printf '\n%*s\n\n' "100" ' ' | tr ' ' "#" | tee -a ${FNAME};
    fi;

    # CPU only
    EXEC_=${EXEC}_nomic;
    FNAME=${RESULT_DIR}/$(basename ${EXEC_})_${fname[k]}.out;
    if [ -f ${EXEC_} ] && [ ! -s ${FNAME} ] ; then
      printf '%*s\n\n' "100" ' ' | tr ' ' "#" | tee -a ${FNAME};
      printf "COMMAND: ${EXEC_} ${args[k]}\n" | tee -a ${FNAME};

      if [[ -n "$PBS_NODEFILE" ]] ;  then 
        ${TIMEOUT} ${max_time[k]} time mpirun -np ${mpi_proc[k]} \
          --hostfile ${PBS_NODEFILE} --bysocket --bind-to-socket  ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      elif [[ -n "$SLURM_NODELIST" ]] ;  then
        ${TIMEOUT} ${max_time[k]} time ibrun tacc_affinity ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      else
        ${TIMEOUT} ${max_time[k]} time mpirun -np ${mpi_proc[k]}  ${EXEC_} ${args[k]} &> >(tee -a ${FNAME});
      fi
      printf '\n%*s\n\n' "100" ' ' | tr ' ' "#" | tee -a ${FNAME};
    fi;

  fi;
done;

