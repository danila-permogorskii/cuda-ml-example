cmake_minimum_required(VERSION 3.18)
project(cuda_ml_example LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architectures for your GPU
# This is for RTX 20xx, 30xx, 40xx series - adjust based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

# Add executable
add_executable(vector_addition src/vector_addition.cu)

# Set properties for CUDA
set_target_properties(vector_addition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(vector_addition ${CUDA_LIBRARIES})

# Add include directories for header files
target_include_directories(vector_addition PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Print some information for debugging
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")