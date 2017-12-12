# MXNetWrapper
MXNet wrapper c++ for loading your network

# Setup
1. Go in src dir
2. Edit the Makefile at line 28 choosing the mxnet installation folder
3. Run make

# The class Wrapper
The class is a simple Wrapper of c_predict_api
It makes easy to load end execute a custom network.

# Example of use
In this example there is an error but I'm struggling to find it!
The example shows a problem of loading a network created by Mathematica.
The network created by mathematica is a simple linear layer with bias.

output = W * input + b

The dimensions are W two rows a 3 column, b two column, input is 3 column and the output is 2 column.

# I'm looking for Help
