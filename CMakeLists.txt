cmake_minimum_required(VERSION 3.17)
 
PROJECT(small-vde)

# ----- Eigen -----
include(FetchContent)
FetchContent_Declare(eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 9441d94dccccd5db8d64179516fdc5b53994a047
        )
FetchContent_Populate(eigen3)
add_library(eigen INTERFACE)
add_library (eigen3::eigen ALIAS eigen)
target_include_directories(eigen INTERFACE ${eigen3_SOURCE_DIR})

add_executable(small-vde MACOSX_BUNDLE main.cpp field.hpp io.hpp vde.hpp)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  MESSAGE(STATUS "Using OpenMP parallelization")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
  MESSAGE(STATUS "Not using OpenMP parallelization")
ENDIF()

target_link_libraries(small-vde eigen)
