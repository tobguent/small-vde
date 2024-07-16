# small-vde #

This folder contains an implementation of 2D case for the paper:

> *Tobias Günther, Holger Theisel*  
**Objective Lagrangian Vortex Cores and Their Visual Representations**  
IEEE Transactions on Visualization and Computer Graphics (IEEE Scientific Visualization 2024), 2025.

This demo code calculates the **Vortex Deviation Error (VDE)** at the example of the cylinder flow.
The implementation was tested on MSVC and GCC. 
A CMake file is provided to compile the program.
The input is the path to the test data set.
The output is a bitmap file that is written into the build folder.

Files:
- `CMakeLists.txt` *Contains the CMake script for cross-platform compilation.*
- `cyl2d.am` *Test data set in the Amira mesh format.*
- `field.hpp` *Class that stores a scalar/vector field and provides suitable getter/setter.*
- `io.hpp` *Collection of helpers for reading Amira files and writing bmp files.*
- `main.cpp` *Contains the entry function of the program.*
- `vde.hpp` *Implementation of VDE.*