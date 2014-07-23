#!/bin/bash

export EXEC=bin/helm

NODES=1;
THRDS=16;
nodes+=(        $NODES );
cores+=(        $THRDS );
mpi_proc+=(     $NODES );
threads+=(      $THRDS );
rho+=(           -2e+2 ); #
ref_tol+=(        1e-4 ); #
min_depth+=(         4 );
max_depth+=(         6 );
fmm_q+=(            14 );
fmm_m+=(            10 );
gmres_tol+=(    1.0e-6 ); #
gmres_iter+=(      400 );
max_time+=(    3600000 );
                                                                                                                       
# Export arrays
export      nodes_="$(declare -p      nodes)";
export      cores_="$(declare -p      cores)";
export   mpi_proc_="$(declare -p   mpi_proc)";
export    threads_="$(declare -p    threads)";
export        rho_="$(declare -p        rho)";
export    ref_tol_="$(declare -p    ref_tol)";
export  min_depth_="$(declare -p  min_depth)";
export  max_depth_="$(declare -p  max_depth)";
export      fmm_q_="$(declare -p      fmm_q)";
export      fmm_m_="$(declare -p      fmm_m)";
export  gmres_tol_="$(declare -p  gmres_tol)";
export gmres_iter_="$(declare -p gmres_iter)";
export   max_time_="$(declare -p   max_time)";

export RESULT_SUBDIR=$(basename ${0%.*});
export WORK_DIR=$(dirname ${PWD}/$0)/..
cd ${WORK_DIR}

TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
./scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

