# MXNetWrapper
MXNet wrapper c++ for loading your network

# Setup
1. Go in src dir
2. Edit the Makefile at line 28 choosing the mxnet installation folder
3. Run make

# The class Wrapper
The class is a simple Wrapper of c_predict_api
It makes easy to load end execute a custom network.

# Example of use src/main.cpp
The example is in the src/main.cpp file.
The example shows a problem of loading a network created and exported by Mathematica.

# Export networks using Mathematica
Due to a bug in the Mathematica export, the network files in the MXNet standard are not totally compliant with the standard MXNet.
To avoid problems both in python or in c++ to load a network generated in Mathematica
you must use the code reported in networks/MXNet_ExportFix.nb in the repository.
