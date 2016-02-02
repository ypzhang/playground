cmake_minimum_required(VERSION 2.8)

project(rsolver)
set (rsolver_VERSION 0.1.0)
set (rsolver_VERSION_MAJOR 0)
set (rsolver_VERSION_MINOR 1)
set (rsolver_VERSION_PATCH 0)

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SparseLib++/1.7/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SparseLib++/1.7/mv/include/)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "'-DCOMPLEX=std::complex<double>'")

#add_definitions(-DCOMPLEX_OSTREAM)
add_definitions('-DCOMPLEX=std::complex<double>')

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")

include (src_list.txt)
add_executable(tgmres testing/tgmres.cc 
	${SPARSE_LIB_SRC_LIST}
	./gsolver/utility.cpp

	)



