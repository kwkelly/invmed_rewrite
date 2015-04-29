#!/bin/bash

cd ../build/
ibrun -np 2 ./test -fmm_m 8 -fmm_q 10 -min_depth 5 -max_depth 5
