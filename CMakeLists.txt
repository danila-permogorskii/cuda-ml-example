cmake_minimum_required(VERSION 3.18)

# Set cmake policy for CUDA integration
# This addresses the CMP0146 warning by explicitly using the new approach
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 NEW)
endif()

# Set the C++ compiler explicitly - adjust path for Windows
# Windows paths use backslashes and include Visual Studio compiler
if(WIN32)
    # For Windows, we'll let CMake find the appropriate compiler
    # But you can uncomment and adjust this if needed
    # set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe")
    # set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin/nvcc.exe")
else()
    # For Linux, use the paths from your example
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

# Set CUDA architecture explicitly for the RTX 3050 Ti
# You can adjust this depending on your GPU
set(CMAKE_CUDA_ARCHITECTURES "86")

# Define the project
project(cuda_ml_example LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)

# Suppress CUDA warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=541")

# Modern CUDA integration - using CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# Add debugging flag for CUDA in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

# Set CUDA optimization flags for Release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Create example directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# Add include directories for header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# List of source files for neural network
set(NEURAL_NETWORK_SOURCES
    src/neural_network.cu
    src/neural_network_kernels.cu
)

# List of source files for text neural network
set(TEXT_NEURAL_NETWORK_SOURCES
    src/text_neural_network.cu
    src/text_neural_network_kernels.cu
    src/text_neural_network_forward.cu
    src/text_neural_network_backward.cu
    src/text_preprocessing.cu
    src/text_neural_network_utils.cu
)

# Add executables
add_executable(vector_addition src/vector_addition.cu)
add_executable(neural_network ${NEURAL_NETWORK_SOURCES} src/main.cu)
add_executable(text_classification ${TEXT_NEURAL_NETWORK_SOURCES} examples/text_classification_demo.cu)

# Set properties for CUDA
set_target_properties(vector_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(neural_network PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(text_classification PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(vector_addition CUDA::cudart)
target_link_libraries(neural_network CUDA::cudart CUDA::curand)
target_link_libraries(text_classification CUDA::cudart CUDA::curand)

# Create directories for models and data if they don't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/models)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)

# Print some information for debugging
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
