cmake_minimum_required(VERSION 3.10)

project(ts_threading_becnhmark)
add_executable(benchmark threading_benchmark.cpp)
target_compile_features(benchmark PRIVATE cxx_std_14)

#
# PyTorch
#

find_package(Torch REQUIRED)
target_include_directories(benchmark PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(benchmark PRIVATE ${TORCH_LIBRARIES})
