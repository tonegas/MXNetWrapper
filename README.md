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
The example shows a problem of loading a network created by Mathematica.

To be confident that my loading wrapper works I have tried to load  Inception-BN network,
and it runs perfectly.

Then I load my simple graph, that is a network created by mathematica with is a simple linear layer with bias.

output = W * input + b

The dimensions are W two rows a 3 column, b two column, input is 3 column and the output is 2 column.

My network has an error but I'm struggling to find it!
I don't know if the problem is in the export or in the configuratiton of the network.
<b>So I hope that some one can help me!</b>
