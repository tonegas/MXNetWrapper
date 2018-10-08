## MXNetWrapper
MXNet c++ wrapper

# Setup
1. Edit the Makefile at line 28 choosing the mxnet installation folder
2. Run make

# The class Wrapper
The class is a simple Wrapper of c_predict_api.
It makes easy to load end execute a custom network.

# Example of use
An example of use is in the test/main.cpp file.
The example shows how to load different networks.

# Export networks from Mathematica
The c++ api of MXNet work slightly differently from the python api,
therefore it needs that all the keys in the binary file .params generated during the export,
start with "arg:" string; meanwhile the api python of MXNet no needs this.
To avoid the loading fail, you need to use an workaround to export a network from
Mathematica (both 11.2 or 11.3). The functions is in MXNetExportFix.11.2.wl for Mathematica 11.2
and MXNetExportFix.11.3.wl for Mathematica 11.3.
