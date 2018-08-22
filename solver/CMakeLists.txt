cmake_minimum_required(VERSION 3.2)

project(rsolver)
set (rsolver_VERSION 0.1.0)
set (rsolver_VERSION_MAJOR 0)
set (rsolver_VERSION_MINOR 1)
set (rsolver_VERSION_PATCH 0)


#find_package(OpenMP)
if (OPENMP_FOUND)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SparseLib++/1.7/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SparseLib++/1.7/mv/include/)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "'-DCOMPLEX=std::complex<double>'")

#add_definitions(-DCOMPLEX_OSTREAM)
add_definitions('-DCOMPLEX=std::complex<double>')

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall")

if (UNIX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -pg") #  -ftree-vectorizer-verbose=2
endif (UNIX)

include (src_list.txt)

add_library(rsolver ${SPARSE_LIB_SRC_LIST}
	./gsolver/utility.cpp
	./gsolver/mmio.cpp
	./gsolver/matrix_reader.cpp
	./gsolver/matrix_convert.cpp
)


add_executable(tgmres testing/tgmres.cc)
target_link_libraries(tgmres rsolver)
add_executable(tbicgsta testing/tbicgsta.cc)
target_link_libraries(tbicgsta rsolver)

# add_subdirectory(testing/catch)





