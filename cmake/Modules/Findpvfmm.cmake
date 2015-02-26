#
# Find pvfmm includes and library
#
# Elemental
# It can be found at:
#
# Elemental_INCLUDE_DIR - where to find pvfmm.h
# Elemental_LIBRARY     - qualified libraries to link against.
# Elemental_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(pvfmm_INCLUDE_DIRS pvfmm.hpp
  $ENV{PVFMM_DIR}/../../include/
  $ENV{PVFMM_DIR}/../../include/pvfmm
  /usr/local/include/pvfmm/include/pvfmm
  /usr/include
  /usr/local/include
)

FIND_LIBRARY(pvfmm_LIBRARY pvfmm
   $ENV{PVFMM_DIR}/../../lib/
   $ENV{PVFMM_DIR}/../../lib/pvfmm
   /usr/local/lib/pvfmm/lib/pvfmm
   /usr/lib/pvfmm/
   /usr/local/lib
   /usr/lib
)

IF(pvfmm_INCLUDE_DIRS AND pvfmm_LIBRARY)
  SET(pvfmm_FOUND "YES")
ENDIF(pvfmm_INCLUDE_DIRS AND pvfmm_LIBRARY)

IF (pvfmm_FOUND)
  IF (NOT pvfmm_FIND_QUIETLY)
    MESSAGE(STATUS
            "Found pvfmm:${pvfmm_LIBRARY}")
  ENDIF (NOT pvfmm_FIND_QUIETLY)
ELSE (pvfmm_FOUND)
  IF (pvfmm_FIND_REQUIRED)
    MESSAGE(STATUS "Elemental not found!")
  ENDIF (pvfmm_FIND_REQUIRED)
ENDIF (pvfmm_FOUND)
