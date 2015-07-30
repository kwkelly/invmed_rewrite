#!/bin/bash

cd ../build/
ibrun -np 1 ./faims -fmm_m 8 -fmm_q 10 -min_depth 3 -max_depth 3 -k 5 -R_d 25 -N_pts 10 -R_s 25
