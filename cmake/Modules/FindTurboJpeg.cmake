# - Find TURBOJPEG
# Find the native TURBOJPEG includes and library
# This module defines
#  TURBOJPEG_INCLUDE_DIR, where to find jpeglib.h, etc.
#  TURBOJPEG_LIBRARIES, the libraries needed to use TURBOJPEG.
#  TURBOJPEG_FOUND, If false, do not try to use TURBOJPEG.
# also defined, but not for general use are
#  TURBOJPEG_LIBRARY, where to find the TURBOJPEG library.

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

find_path(TURBOJPEG_INCLUDE_DIR turbojpeg.h)

set(TURBOJPEG_NAMES ${TURBOJPEG_NAMES} turbojpeg)
find_library(TURBOJPEG_LIBRARY NAMES ${TURBOJPEG_NAMES} )

# handle the QUIETLY and REQUIRED arguments and set TURBOJPEG_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TurboJpeg DEFAULT_MSG TURBOJPEG_LIBRARY TURBOJPEG_INCLUDE_DIR)

if(TURBOJPEG_FOUND)
  set(TURBOJPEG_LIBRARIES ${TURBOJPEG_LIBRARY})
endif()

# Deprecated declarations.
set (NATIVE_TURBOJPEG_INCLUDE_PATH ${TURBOJPEG_INCLUDE_DIR} )
if(TURBOJPEG_LIBRARY)
  get_filename_component (NATIVE_TURBOJPEG_LIB_PATH ${TURBOJPEG_LIBRARY} PATH)
endif()

mark_as_advanced(TURBOJPEG_LIBRARY TURBOJPEG_INCLUDE_DIR )