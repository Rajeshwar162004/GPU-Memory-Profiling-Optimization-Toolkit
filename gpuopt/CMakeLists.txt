cmake_minimum_required(VERSION 3.18)
project(GPUOpt_Kernels LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA architecture - adjust for your GPU
# Common architectures: 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere), 89 (Ada), 90 (Hopper)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures to build for")

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Vector Add
add_executable(vector_add_naive kernels/vector_add/naive.cu)
target_link_libraries(vector_add_naive ${CUDA_LIBRARIES})
set_target_properties(vector_add_naive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(vector_add_optimized kernels/vector_add/optimized.cu)
target_link_libraries(vector_add_optimized ${CUDA_LIBRARIES})
set_target_properties(vector_add_optimized PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Strided Access
add_executable(strided_access kernels/strided_access/strided.cu)
target_link_libraries(strided_access ${CUDA_LIBRARIES})
set_target_properties(strided_access PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Transpose
add_executable(transpose kernels/transpose/transpose.cu)
target_link_libraries(transpose ${CUDA_LIBRARIES})
set_target_properties(transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Matrix Multiplication
add_executable(matmul kernels/matmul/matmul.cu)
target_link_libraries(matmul ${CUDA_LIBRARIES})
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Reduction
add_executable(reduction kernels/reduction/reduction.cu)
target_link_libraries(reduction ${CUDA_LIBRARIES})
set_target_properties(reduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Install targets (optional)
install(TARGETS vector_add_naive vector_add_optimized strided_access transpose matmul reduction
        RUNTIME DESTINATION bin)

# Add custom target to build all
add_custom_target(build_all DEPENDS vector_add_naive vector_add_optimized strided_access transpose matmul reduction)

# Print configuration
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")