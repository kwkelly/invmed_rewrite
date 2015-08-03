set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -mavx -wd3218 -wd2570 -no-offload -Qoption,cpp,--extended_float_type -openmp -no-offload -DALLTOALLV_FIX" CACHE STRING "" FORCE)
