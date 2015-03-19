cmake_minimum_required (VERSION 2.6)
project (wstFD)

set (FFTW "/Users/wsttiger/local/fftw-3.3.4-icc")
set (CMAKE_CXX_FLAGS "-g -O1 -std=c++11 -mkl")

include_directories("${FFTW}/include")

add_executable(cos1d cos1d.cc)
