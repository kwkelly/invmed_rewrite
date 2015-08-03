cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(InvMed CXX)

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

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
#find_package(PETSc REQUIRED)
#include_directories(${PETSC_INCLUDES})
find_package(pvfmm REQUIRED)
include_directories(${pvfmm_INCLUDE_DIRS})
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
find_package(Elemental REQUIRED)
include_directories(${Elemental_INCLUDE_DIRS})

include_directories("include")
include_directories("extern/rsvd/include")


add_executable(invmed src/invmed.cpp)
#add_executable(test test/test.cpp)
add_executable(faims test/faims.cpp)
add_executable(dist_test test/dist_test.cpp)
add_executable(faims_tests test/faims_tests.cpp)
#add_executable(empty_test src/empty_test.cpp)
add_library(point_distribs src/point_distribs.cpp)
#add_library(helm_kernels src/helm_kernels.cpp)
add_library(funcs src/funcs.cpp)
add_library(operators src/operators.cpp)
add_library(convert_elemental src/convert_elemental.cpp)
#add_library(convert_petsc src/convert_petsc.cpp)


message(STATUS ${pvfmm_INCLUDE_DIRS})
message(STATUS ${MPI_CXX_LIBRARIES})

set(LIBS ${LIBS} ${pvfmm_LIBRARY} ${Elemental_LIBRARIES} ${MPI_CXX_LIBRARIES} point_distribs funcs convert_elemental operators)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")

#target_link_libraries(invmed ${LIBS})
#target_link_libraries(test ${LIBS})
#target_link_libraries(dist_test ${LIBS})
#target_link_libraries(faims ${LIBS})
target_link_libraries(faims_tests ${LIBS})
#target_link_libraries(empty_test ${LIBS})
