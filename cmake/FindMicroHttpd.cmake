# -*- mode: cmake; -*-
# - Try to find libmicrohttpd include dirs and libraries
# Usage of this module as follows:
# This file defines:
# * MICROHTTPD_FOUND if protoc was found
# * MICROHTTPD_LIBRARY The lib to link to (currently only a static unix lib, not portable)
# * MICROHTTPD_INCLUDE The include directories for libmicrohttpd.

#message(STATUS "FindMicrohttpd check")
# Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
if(MICROHTTPD_USE_STATIC_LIBS)
  set(_microhttpd_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a )
  endif()
endif()
IF(NOT WIN32)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(_MICROHTTPD QUIET microhttpd>=0.9)
        if("" STREQUAL "${_MICROHTTPD_CFLAGS_OTHER}")
            pkg_check_modules(_MICROHTTPD QUIET libmicrohttpd>=0.9)
        endif()
        set(MICROHTTPD_DEFINITIONS ${_MICROHTTPD_CFLAGS_OTHER})
    endif()
endif()

#
# set defaults
SET(_microhttpd_INCLUDE_SEARCH_DIRS
    ${CMAKE_INCLUDE_PATH}
    /usr/local/include
    /usr/include
    )

SET(_microhttpd_LIBRARIES_SEARCH_DIRS
    ${CMAKE_LIBRARY_PATH}
    /usr/local/lib
    /usr/lib
    )

##
if(NOT "" MATCHES "$ENV{MICROHTTPD_HOME}")
    set (MICROHTTPD_HOME "$ENV{MICROHTTPD_HOME}")
endif()
IF( NOT $ENV{MICROHTTPD_INCLUDEDIR} STREQUAL "" )
    SET(_microhttpd_INCLUDE_SEARCH_DIRS $ENV{MICROHTTPD_INCLUDEDIR} ${_microhttpd_INCLUDE_SEARCH_DIRS})
ENDIF( NOT $ENV{MICROHTTPD_INCLUDEDIR} STREQUAL "" )
IF( NOT $ENV{MICROHTTPD_LIBRARYDIR} STREQUAL "" )
    SET(_microhttpd_LIBRARIES_SEARCH_DIRS $ENV{MICROHTTPD_LIBRARYDIR} ${_microhttpd_LIBRARIES_SEARCH_DIRS})
ENDIF( NOT $ENV{MICROHTTPD_LIBRARYDIR} STREQUAL "" )

##

IF( NOT ${MICROHTTPD_HOME} STREQUAL "" )
    #message(STATUS "Looking for microhttpd in ${MICROHTTPD_HOME}")
    set(_microhttpd_INCLUDE_SEARCH_DIRS ${MICROHTTPD_HOME}/include ${_microhttpd_INCLUDE_SEARCH_DIRS})
    set(_microhttpd_LIBRARIES_SEARCH_DIRS ${MICROHTTPD_HOME}/lib ${_microhttpd_LIBRARIES_SEARCH_DIRS})
    set(_microhttpd_HOME ${MICROHTTPD_HOME})
ENDIF( NOT ${MICROHTTPD_HOME} STREQUAL "" )

IF( MICROHTTPD_HOME )
    SET(_microhttpd_INCLUDE_SEARCH_DIRS ${MICROHTTPD_HOME}/include ${_microhttpd_INCLUDE_SEARCH_DIRS})
    SET(_microhttpd_LIBRARIES_SEARCH_DIRS ${MICROHTTPD_HOME}/lib ${_microhttpd_LIBRARIES_SEARCH_DIRS})
    SET(_microhttpd_HOME ${MICROHTTPD_HOME})
ENDIF( MICROHTTPD_HOME )

# find the include files
FIND_PATH(MICROHTTPD_INCLUDE_DIR microhttpd.h
    HINTS
        ${_microhttpd_INCLUDE_SEARCH_DIRS}
        ${_MICROHTTPD_INCLUDEDIR}
        ${_MICROHTTPD_INCLUDE_DIRS}
        ${CMAKE_INCLUDE_PATH}
)

# locate the library
IF(WIN32)
    SET(MICROHTTPD_LIBRARY_NAMES ${MICROHTTPD_LIBRARY_NAMES} libmicrohttpd.lib)
ELSE(WIN32)
    SET(MICROHTTPD_LIBRARY_NAMES ${MICROHTTPD_LIBRARY_NAMES} libmicrohttpd.so)
    #SET(MICROHTTPD_LIBRARY_NAMES ${MICROHTTPD_LIBRARY_NAMES} libmicrohttpd.a)
ENDIF(WIN32)
FIND_LIBRARY(MICROHTTPD_LIBRARY NAMES ${MICROHTTPD_LIBRARY_NAMES}
    HINTS
        ${_microhttpd_LIBRARIES_SEARCH_DIRS}
        ${_MICROHTTPD_LIBDIR}
        ${_MICROHTTPD_LIBRARY_DIRS}
)

# if the include and the program are found then we have it
IF(MICROHTTPD_INCLUDE_DIR AND MICROHTTPD_LIBRARY)
    SET(MICROHTTPD_FOUND TRUE)
    set(MICROHTTPD_VERSION ${_MICROHTTPD_VERSION})
    if(NOT WIN32)
        set(MICROHTTPD_LIBS_EXTRA "-lrt")
    endif(NOT WIN32)
    set(MICROHTTPD_LIBRARIES ${MICROHTTPD_LIBRARY})
    list(APPEND MICROHTTPD_LIBRARIES "${MICROHTTPD_LIBS_EXTRA}")
ENDIF(MICROHTTPD_INCLUDE_DIR AND MICROHTTPD_LIBRARY)

MARK_AS_ADVANCED(
    MICROHTTPD_FOUND
    MICROHTTPD_VERSION
    MICROHTTPD_LIBRARY
    MICROHTTPD_LIBS_EXTRA
    MICROHTTPD_LIBRARIES
    MICROHTTPD_INCLUDE_DIR
)

# Restore the original find library ordering
if(MICROHTTPD_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_microhttpd_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()
# vim: et sw=4 sts=4 ts=4:
