cmake_minimum_required(VERSION 3.18)

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

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add debugging flag for CUDA in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

# Set CUDA optimization flags for Release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Add include directories for header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the executable
add_executable(vector_addition src/vector_addition.cu)

# Set properties for CUDA
set_target_properties(vector_addition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(vector_addition ${CUDA_LIBRARIES})

# Print some information for debugging
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
