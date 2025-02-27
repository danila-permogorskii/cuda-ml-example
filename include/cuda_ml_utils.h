#ifndef CUDA_ML_UTILS_H
#define CUDA_ML_UTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

// Utility function to check CUDA errors
inline void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d: %s\n", file, line, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(val) checkCuda((val), __FILE__, __LINE__)

// Get CUDA device properties and print them
inline void printDeviceInfo() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(error));
        return;
    }
    
    printf("Found %d CUDA devices\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("\nDevice %d: \"%s\"\n", i, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               static_cast<float>(deviceProp.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Multiprocessor Count: %d\n", deviceProp.multiProcessorCount);
        printf("  Max Threads Dimensions: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Size: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
}

// Calculate optimal thread and block counts for CUDA kernel launch
inline void calculateOptimalLaunchParams(int dataSize, int &blocks, int &threads) {
    // Get device properties
    cudaDeviceProp prop;
    int deviceId;
    
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);
    
    // Standard practice is to use a power of 2 for thread count
    threads = 256; // Common thread count, adjust based on kernel needs
    
    // Calculate blocks needed for the data size
    blocks = (dataSize + threads - 1) / threads;
    
    // Cap the number of blocks if needed to avoid excessive blocks
    int maxBlocks = prop.multiProcessorCount * 32; // Arbitrary multiplier, can be tuned
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }
}

// Timer utility for CUDA operations
class CudaTimer {
private:
    cudaEvent_t start, stop;
    bool initialized;

public:
    CudaTimer() : initialized(false) {}
    
    void initialize() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        initialized = true;
    }
    
    void startTimer() {
        if (!initialized) initialize();
        cudaEventRecord(start, 0);
    }
    
    float stopTimer() {
        float time = 0.0f;
        if (!initialized) return time;
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        return time; // Returns time in milliseconds
    }
    
    ~CudaTimer() {
        if (initialized) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
};

#endif // CUDA_ML_UTILS_H