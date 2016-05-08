cmake_minimum_required(VERSION 3.2)
project(gpu_solver)
set (gpu_solver_VERSION 0.1.0)
find_package(CUDA REQUIRED)
# CUDA architecture options, choose what you want
set(GENCODE_SM30 "-gencode=arch=compute_30,code=sm_30")
set(GENCODE_SM35 "-gencode=arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${GENCODE_SM30} ${GENCODE_SM35})
# Add the base directory to your include file search path
include_directories(${CMAKE_SRC_DIR})
set(SRC_LIST
	src/auxil.cpp  
	src/blusol.cu
	src/ds.cpp
	src/gmres.cpp
	src/iluk.cpp 
	src/io.cpp
	src/lspolyn.cpp
	src/mem.cu
	src/mmio.h      
	src/precon.cpp  
	src/rcm.cpp         
	#skit.f   
	src/texture.h
	src/biluk.cpp
	src/datatype.h
	src/format.cpp
	src/gpusollib.h  
	src/ilut.cpp
	src/lanczos.cpp  
	src/mcilu0.cu    
	src/mmd.cpp  
	src/pcg.cpp     
	src/protos.h    
	src/reorder.cpp     
	src/spmv.cu  
	src/triangsol.cu
	src/bilut.cpp  
	src/domdecomp.cpp  
	src/ggexp.cpp   
	src/ic.cpp       
	src/interface-hui.cpp  
	src/level.cpp    
	src/mcsor.cpp    
	src/mmio.c   
	src/pitsol.cpp  
	src/rcm.H       
	src/reorder_mc.cpp  
	src/ssor.cu  
	src/utils.cu
)

#if it is a shared library: (if static, change SHARED to STATIC)
cuda_add_library(gpu_solver SHARED ${SRC_LIST})
# or if it is an executable
# cuda_add_executable(your_project_name ${SRC_LIST})
