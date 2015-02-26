cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER mpicc)

project(InvMed CXX C)

function(echo_all_cmake_variable_values)
  message(STATUS "")
  get_cmake_property(vs VARIABLES)
  foreach(v ${vs})
    message(STATUS "${v}='${${v}}'")
  endforeach(v)
  message(STATUS "")
endfunction()


# Read in all the variables from PVFMM MakeVariables
file(STRINGS $ENV{PVFMM_DIR}/MakeVariables make_variables)

#foreach(line ${make_variables})
#  string(STRIP ${line} line1)
#  string(FIND ${line1} "#" loc)
#  if(${loc} EQUAL -1)
#    string(REPLACE "=" ";" line_list ${line1})
#    list(LENGTH line_list len)
#    list(GET line_list 0 var_name)
#    if(len==1)
#      string(STRIP ${var_name} var_name1)
#      set(${var_name1} "")
#    else(len==1)
#      list(GET line_list 1 var_value)
#    endif(len==1)
#    string(STRIP ${var_name} var_name1)
#    string(STRIP ${var_value} var_value1)
#    set(${var_name1} ${var_value1})
#  endif(${loc} EQUAL -1)
#endforeach(line)

#echo_all_cmake_variable_values()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

add_executable(invmed src/invmed.cpp src/funcs.cpp)
add_executable(test test/test.cpp src/funcs.cpp)

find_package(PETSc REQUIRED)
include_directories(${PETSC_INCLUDES})
find_package(pvfmm REQUIRED)
include_directories(${pvfmm_INCLUDE_DIRS})
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
find_package(Elemental REQUIRED)
include_directories(${Elemental_INCLUDE_DIRS})

include_directories("include")

message(STATUS ${pvfmm_INCLUDE_DIRS})

set(LIBS ${LIBS} ${PETSC_LIBRARIES} ${pvfmm_LIBRARY} ${Elemental_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-offload -O2 -wd3218 -wd2570 -DFFTW3_MKL -Qoption,cpp,--extended_float_type -openmp -no-offload -DALLTOALLV_FIX -debug -O0 -std=c++11 -ldl -lstdc++ -limf -mkl")

target_link_libraries(invmed ${LIBS})
target_link_libraries(test ${LIBS})
