#
# Find Elemental includes and library
#
# Elemental
# It can be found at:
#
# Elemental_INCLUDE_DIR - where to find El.hpp
# Elemental_LIBRARIES     - qualified libraries to link against.
# Elemental_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(Elemental_INCLUDE_DIRS El.hpp
  /usr/local/include
  /usr/include
  $ENV{HOME}/Software/include
  $ENV{ELEMENTAL_DIR}/include
)

FIND_LIBRARY(Elemental_LIBRARIES El
   /usr/local/lib
   /usr/lib
   $ENV{HOME}/Software/lib
   $ENV{ELEMENTAL_DIR}/lib
)

FIND_LIBRARY(Pmrrr_LIBRARY pmrrr
   /usr/local/lib
   /usr/lib
   $ENV{HOME}/Software/lib
   $ENV{ELEMENTAL_DIR}/lib
)

IF(Elemental_INCLUDE_DIRS AND Elemental_LIBRARIES AND Pmrrr_LIBRARY)
  SET(Elemental_FOUND "YES")
ENDIF(Elemental_INCLUDE_DIRS AND Elemental_LIBRARIES AND Pmrrr_LIBRARY)

IF (Elemental_FOUND)
  IF (NOT Elemental_FIND_QUIETLY)
    MESSAGE(STATUS
            "Found Elemental:${Elemental_LIBRARIES};${Elemental_LIBRARIES_C}")
    MESSAGE(STATUS
            "Found pmrrr:${Pmrrr_LIBRARY}")
  ENDIF (NOT Elemental_FIND_QUIETLY)
ELSE (Elemental_FOUND)
  IF (Elemental_FIND_REQUIRED)
    MESSAGE(STATUS "Elemental not found!")
  ENDIF (Elemental_FIND_REQUIRED)
ENDIF (Elemental_FOUND)
