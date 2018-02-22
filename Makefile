#
# Authors: Gastone Pietro Rosati Papini
#
# Use "make all" to build the load network functions
#
# The load network functions bin is generated in this folder
#

$(info ======================= network loader =========================)

# Check OS
OS = $(shell uname)

ifeq ($(OS), Darwin)
$(info OS detected -> MAC OS X)
else ifeq ($(OS), Linux)
$(info OS detected -> LINUX)
else
$(error OS detected $(OS) is NOT SUPPORTED)
endif

$(info ================================================================)

# Debug Variable
ENABLE_DEBUG = -DDEBUG=1

# MXNet
MXNET_DIR = incubator-mxnet
MXNET_INCLUDE_DIR = $(MXNET_DIR)/include/
MXNET_LIB = $(MXNET_DIR)/lib/libmxnet.so

# Network loader files
MXWRAPPER_DIR = src
MXWRAPPER_SRCS = $(MXWRAPPER_DIR)/MXNetWrapper.cc
MXWRAPPER_H = $(MXWRAPPER_DIR)/MXNetWrapper.h

#Commands
CD = cd

# Define the compiler to use
CC = clang++

ifeq ($(OS), Darwin)
ARCHFLAGS = -arch x86_64
else ifeq ($(OS), Linux)
ARCHFLAGS =
endif

# Define any compile-time flags
CFLAGS = -std=c++11 -O2 -DNOT_INLINED=1 -DUSE_CODRIVER=1

# Define library paths in addition to /usr/lib
LFLAGS = -L$(MXNET_LIB)

# Output directory
TEST_DIR = test

# LINKER flags
ifeq ($(OS), Darwin)
LDFLAGS = -stdlib=libc++ -std=gnu++11
else ifeq ($(OS), Linux)
LDFLAGS = -stdlib=libstdc++
endif

# Define any libraries to link into executable
ifeq ($(OS), Darwin)
LIBS = -llm
else ifeq ($(OS), Linux)
LIBS = -llm
endif

# Define any directories containing header files other than /usr/include
INCLUDES = -I$(MXNET_INCLUDE_DIR) -I$(MXWRAPPER_DIR)

# Define the source files
SRCS =  $(MXWRAPPER_SRCS)

# Define the object files
OBJS = $(SRCS:.cc=.o)

all: $(OBJS)

mxnet:
	$(CD) $(MXNET_DIR); make -j $(nproc) USE_OPENCV=0

test: $(OBJS)
	$(CC) $(ARCHFLAGS) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(ENABLE_DEBUG) $(TEST_DIR)/main.cpp -o $(TEST_DIR)/test $(OBJS) $(LFLAGS) $(MXNET_LIB)

.cc.o:
	$(CC) $(ARCHFLAGS) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(ENABLE_DEBUG) -c $<  -o $@

clean:
	$(RM) $(OBJS) $(TEST_DIR)/test
