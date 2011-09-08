#-------------------------------------------------------------------------------
#
# Copyright 2010-2011 Duane Merrill
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
#  
#  For more information, see our Google Code project site: 
#  http://code.google.com/p/back40computing/
#  
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
# Build script for project
#-------------------------------------------------------------------------------

NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

KERNELS = 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
#SM_TARGETS = $(GEN_SM20) $(GEN_SM13) $(GEN_SM10)
SM_TARGETS = $(GEN_SM20) 

#-------------------------------------------------------------------------------
# Tune arch
#-------------------------------------------------------------------------------

ifdef tunearch
    TUNE_ARCH = $(tunearch)
else 
	TUNE_ARCH = 200
endif

ifeq ($(TUNE_ARCH), 200) 
	TUNE_SM_TARGETS = $(GEN_SM20)
endif
ifeq ($(TUNE_ARCH), 130) 
	TUNE_SM_TARGETS = $(GEN_SM13)
endif
ifeq ($(TUNE_ARCH), 100) 
	TUNE_SM_TARGETS = $(GEN_SM10)
endif

#-------------------------------------------------------------------------------
# Tune size
#-------------------------------------------------------------------------------

ifdef tunesize
    TUNE_SIZE = $(tunesize)
else 
	TUNE_SIZE = 4
endif


#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

CUDA_INC = "$(shell dirname $(NVCC))/../include"
INC = -I$(CUDA_INC) -I.. -I../.. 

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES = 

#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

ifneq ($(force64), 1)
	# Compile with 32-bit device pointers by default
	ARCH_SUFFIX = i386
	ARCH = -m32
else
	ARCH_SUFFIX = x86_64
	ARCH = -m64
endif

NVCCFLAGS = -Xptxas -v -Xcudafe -\#

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
endif

ifeq (,$(findstring 3.0, $(NVCC_VERSION)))
ifneq ($(abi), 1)
	# Disable the ABI by default for 3.1+
	NVCCFLAGS += -Xptxas -abi=no
endif
endif

ifeq ($(verbose), 1)
    NVCCFLAGS += -v
endif

ifeq ($(keep), 1)
    NVCCFLAGS += -keep
endif

ifdef maxregisters
    NVCCFLAGS += -maxrregcount $(maxregisters)
endif

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS = 			./Makefile \
				./test_segmented_scan.h \
				../b40c_test_util.h \
				$(wildcard ../../b40c/*.cuh) \
				$(wildcard ../../b40c/util/*.cuh) \
				$(wildcard ../../b40c/util/io/*.cuh) \
				$(wildcard ../../b40c/util/reduction/*.cuh) \
				$(wildcard ../../b40c/util/scan/*.cuh) \
				$(wildcard ../../b40c/util/reduction/soa/*.cuh) \
				$(wildcard ../../b40c/util/scan/soa/*.cuh) \
				$(wildcard ../../b40c/segmented_scan/*.cuh) \
				$(wildcard ../../b40c/reduction/*.cuh) \
				$(wildcard ../../b40c/scan/*.cuh) 
				 


#-------------------------------------------------------------------------------
# (make simple) Simple example driver for segmented scan
#-------------------------------------------------------------------------------

simple: bin/simple_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/simple_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX) : simple_segmented_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/simple_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX) simple_segmented_scan.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  



#-------------------------------------------------------------------------------
# (make test) Test driver 
#-------------------------------------------------------------------------------

test: bin/test_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_segmented_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_segmented_scan.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  

	
#-------------------------------------------------------------------------------
# (make tune) Evaluates the space of configuration types
#-------------------------------------------------------------------------------

tune: bin/tune_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX)_sm$(TUNE_ARCH)_u$(TUNE_SIZE)B

bin/tune_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX)_sm$(TUNE_ARCH)_u$(TUNE_SIZE)B : tune_segmented_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(TUNE_SM_TARGETS) -o bin/tune_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX)_sm$(TUNE_ARCH)_u$(TUNE_SIZE)B tune_segmented_scan.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3 -DTUNE_ARCH=$(TUNE_ARCH) -DTUNE_SIZE=$(TUNE_SIZE) 


#-------------------------------------------------------------------------------
# (make compare) Compares performance against other libraries (e.g., Thrust)
#-------------------------------------------------------------------------------

compare: bin/compare_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/compare_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX) : compare_segmented_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/compare_segmented_scan_$(NVCC_VERSION)_$(ARCH_SUFFIX) compare_segmented_scan.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  


#
# Clean
#

clean :
	rm -f bin/*_$(NVCC_VERSION)_$(ARCH_SUFFIX)* 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o




