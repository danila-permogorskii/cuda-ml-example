# CUDA ML Example Project

A simple CUDA project demonstrating basic machine learning operations using NVIDIA's CUDA platform.

## Project Overview

This project includes:
- Vector addition implementation (a fundamental operation in ML)
- Element-wise multiplication (used in various ML algorithms)
- ReLU activation function (common in neural networks)
- Utility functions for CUDA development

## Prerequisites

To build and run this project, you need:

1. Windows 11
2. NVIDIA GPU with CUDA support
3. CUDA Toolkit (recommended version 11.0 or higher)
4. CLion IDE
5. CMake (version 3.18 or higher)
6. A C++ compiler compatible with CUDA (e.g., MSVC)

## Setup Instructions

### 1. Install CUDA Toolkit

1. Download and install the latest CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
2. Make sure to select the correct version for your Windows 11 system.
3. Follow the installation wizard and install in the default location.
4. The installer will also install the necessary GPU drivers if not already present.

### 2. Configure CLion

1. Open CLion
2. Go to Settings/Preferences (Ctrl+Alt+S)
3. Navigate to Build, Execution, Deployment > Toolchains
4. Make sure your Visual Studio toolchain is set as the default
5. CLion should automatically detect the CUDA toolkit if installed properly

### 3. Build and Run the Project

1. Create a new directory structure matching the project files:
   ```
   cuda_ml_example/
   ├── CMakeLists.txt
   ├── include/
   │   └── cuda_ml_utils.h
   └── src/
       └── vector_addition.cu
   ```

2. Copy the provided files into this directory structure
3. Open the project in CLion
4. Click the Build button (or press Ctrl+F9)
5. Once built successfully, click the Run button (or press Shift+F10)

### 4. Troubleshooting

If you encounter build issues:

1. **CMake cannot find CUDA:**
   - Make sure CUDA is installed correctly
   - Check if the CUDA bin directory is in your PATH environment variable
   - Try restarting CLion

2. **Compilation errors:**
   - Verify you have a compatible GPU
   - Check that you're using a CUDA-compatible version of the compiler

3. **Runtime errors:**
   - Check that your GPU supports the CUDA compute capability set in CMakeLists.txt
   - You may need to adjust `CMAKE_CUDA_ARCHITECTURES` to match your GPU

## Customizing for Your GPU

In the `CMakeLists.txt` file, find the line:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 86)
```

Adjust this based on your NVIDIA GPU:
- 75 - RTX 20xx series
- 86 - RTX 30xx series
- 89 - RTX 40xx series

You can find your GPU's compute capability in the [CUDA GPU Computing Capability table](https://developer.nvidia.com/cuda-gpus).

## Extending the Project

This basic project can be extended in several ways:

1. Implement more complex ML operations like convolutions
2. Add matrix multiplication (crucial for neural networks)
3. Implement a simple neural network layer
4. Add batched processing capabilities
5. Implement gradient descent algorithm

## Performance Optimization

To optimize CUDA performance:
1. Use the CudaTimer utility to benchmark different implementations
2. Experiment with different thread/block configurations
3. Consider using shared memory for frequently accessed data
4. Minimize data transfers between host and device