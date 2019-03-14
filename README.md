### MXNetWrapper
MXNet c++ wrapper

## Setup
1. Edit the Makefile at line 28 choosing the mxnet installation folder
2. Run make

## The class Wrapper
The class is a simple Wrapper of c_predict_api.
It makes easy to load end execute a custom network.

## Example of use
An example of use is in the test/main.cpp file.
The example shows how to load different networks.

## Export networks from Mathematica
The c++ api of MXNet work slightly differently from the python api,
therefore it needs that all the keys in the binary file .params generated during the export,
start with "arg:" string; meanwhile the api python of MXNet no needs this.
To avoid the loading fail, you need to use an workaround to export a network from
Mathematica (both 11.2 or 11.3). The functions is in MXNetExportFix.11.2.wl for Mathematica 11.2
and MXNetExportFix.11.3.wl for Mathematica 11.3.

## Export networks from Python
# Presetup
1. sudo apt get install python3
2. sudo apt install virtualenv
3. virtualenv --python=/usr/bin/python3 .mxnet
4. source .mxnet/bin/activate
5. pip3 install --upgrade pip
6. pip install ipython
7. pip install jupyter
8. pip install keras keras-mxnet
# How to create Network
1. open the file test-export-python.ipynb and follow the code
