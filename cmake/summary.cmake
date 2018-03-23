# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

function(report_status text)
    set(status_cond)
    set(status_then)
    set(status_else)
    set(status_current_name "cond")
    foreach(arg ${ARGN})
        if(arg STREQUAL "THEN")
            set(status_current_name "then")
        elseif(arg STREQUAL "ELSE")
            set(status_current_name "else")
        else()
            list(APPEND status_${status_current_name} ${arg})
        endif()
    endforeach()
    if (DEFINED status_cond)
        if (DEFINED status_then OR DEFINED status_else)
            if (${status_cond})
                string(REPLACE ";" " " status_then "${status_then}")
                string(REGEX REPLACE "^[ \t]+" "" status_then "${status_then}")
                message("${text}${status_then}")
            else()
                string(REPLACE ";" " " status_else "${status_else}")
                string(REGEX REPLACE "^[ \t]+" "" status_else "${status_else}")
                message("${text}${status_else}")
            endif()
        else()
            string(REPLACE ";" " " status_cond "${status_cond}")
            string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
            message("${text}${status_cond}")
        endif()
    else()
        message("${text}")
    endif()
endfunction()

function(print_configuration_summary)
    set(AEON_VERSION ${AEON_VERSON} PARENT_SCOPE)
    report_status("")
    report_status("===[ Build Summary ]=============================================================")
    report_status("")
    report_status("  General:")
    report_status("    Version                    : ${AEON_VERSION}")
    report_status("    System                     : ${DISTRIB_DESCRIPTION} (${CMAKE_SYSTEM_NAME})")
    report_status("    C++ Compiler               : ${CMAKE_CXX_COMPILER} (ver. ${CMAKE_CXX_COMPILER_VERSION})")
    report_status("    Build Type                 : ${CMAKE_BUILD_TYPE}")
    report_status("")
    report_status("    AEON Library               : Yes")
    if (CPPREST_FOUND)
        report_status("    AEON Service               : " ENABLE_AEON_SERVICE THEN "Yes" ELSE "Disabled")
    else()
        report_status("    AEON Service               : No")
    endif()
    report_status("    AEON Client                : " ENABLE_AEON_CLIENT THEN "Yes" ELSE "No")
    report_status("    AEON Python Interface      : " PYTHONLIBS_FOUND AND NUMPY_FOUND THEN "Yes" ELSE "No (Python not found, see details below)")
    report_status("    AEON Python Plugin         : " PYTHON_PLUGIN THEN "Yes" ELSE "No")
    if (ENABLE_AEON_SERVICE)
    report_status("")
    report_status("  Service Connectors:")
        report_status("    RESTfull                   : Yes")
        if (OPENFABRICS_FOUND)
            report_status("    RDMA                       : " ENABLE_OPENFABRICS_CONNECTOR THEN "Yes" ELSE "Disabled")
        else()
            report_status("    RDMA                       : No")
        endif()
    endif()
    report_status("")
    report_status("  Dependencies:")
    report_status("    Boost                      : Yes (ver. ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION})")
    report_status("    OpenCV                     : " OpenCV_FOUND THEN "Yes (ver. ${OpenCV_VERSION})" ELSE "Not Found")
    report_status("    SoX                        : " SOX_FOUND THEN "Yes (ver. ${SOX_VERSION})" ELSE "Not Found")
    report_status("    Curl                       : " CURL_FOUND THEN "Yes (ver. ${CURL_VERSION_STRING})" ELSE "Not Found")
    report_status("    Nlohmann::Json             : Yes (ver. ${NLOHMANN_JSON_VERSION})")
    if (ENABLE_AEON_SERVICE)
        report_status("    CppREST                    : " CPPREST_FOUND THEN "Yes (ver. ${CPPREST_VERSION})" ELSE "Not Found")
    endif()
    if (ENABLE_OPENFABRICS_CONNECTOR)
        report_status("    Fabric                     : " OPENFABRICS_FOUND THEN "Yes (ver. ${OPENFABRICS_VERSION})" ELSE "Not Found")
    endif()
    if (ENABLE_AEON_SERVICE)
        report_status("    OpenSSL                    : " OPENSSL_FOUND THEN "Yes (ver. ${OPENSSL_VERSION})" ELSE "Not Found")
    endif()
    report_status("")
    report_status("  Python:")
    report_status("    Interpreter                : " PYTHON_EXECUTABLE THEN "${PYTHON_EXECUTABLE} (ver. ${PYTHON_VERSION_STRING})" ELSE "No")
    report_status("    Libraries                  : " PYTHONLIBS_FOUND THEN "${PYTHON_LIBRARIES} (ver. ${PYTHONLIBS_VERSION_STRING})" ELSE "No")
    report_status("    NumPy                      : " NUMPY_FOUND THEN "${NUMPY_INCLUDE_DIRS} (ver. ${NUMPY_VERSION})" ELSE "No")
    report_status("")
    report_status("  Documentation:")
    report_status("    Doxygen                    : " DOXYGEN_FOUND THEN "${DOXYGEN_EXECUTABLE} (ver. ${DOXYGEN_VERSION})" ELSE "No")
    if (DOXYGEN_FOUND)
        report_status("    Doxygen config             : ${CMAKE_CURRENT_BINARY_DIR}/doxygen.conf")
    endif()
    report_status("    LaTeX                      : " LATEX_FOUND THEN "${LATEX_COMPILER}" ELSE "No")
    report_status("    Sphinx                     : " SPHINX_FOUND THEN "${SPHINX_EXECUTABLE} (ver. ${SPHINX_VERSION})" ELSE "No")
    if (SPHINX_FOUND)
        report_status("    Sphinx config              : ${CMAKE_CURRENT_BINARY_DIR}/source/conf.py")
    endif()
    if (BREATHE_MISSING_REQUIREMENTS)
        report_status("    Breathe                    : No (Requires ${BREATHE_MISSING_REQUIREMENTS})")
    else()
        report_status("    Breathe                    : " BREATHE_FOUND THEN "${BREATHE_EXECUTABLE} (ver. ${BREATHE_VERSION})" ELSE "No")
    endif()
    report_status("")
    report_status("Install:")
    report_status("    Installation path          : ${CMAKE_INSTALL_PREFIX}")
    report_status("")
    report_status("================================================================================")
    report_status("")
endfunction()
