#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
// Each thread performs one addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    // Calculate global thread ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Check if within array bounds
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// CUDA kernel for vector element-wise multiplication (used in many ML algorithms)
__global__ void vectorMultiply(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements) {
        C[i] = A[i] * B[i];
    }
}

// CUDA kernel for ReLU activation function (common in neural networks)
__global__ void reluActivation(float *data, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements) {
        // ReLU activation: max(0, x)
        data[i] = fmaxf(0.0f, data[i]);
    }
}

// Function to check errors in CUDA calls
#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

int main(void) {
    // Print basic info
    printf("Vector Addition and Basic ML Operations with CUDA\n");
    
    // Error code for checking CUDA operations
    cudaError_t err = cudaSuccess;
    
    // Array size and number of bytes
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("Vector size: %d\n", numElements);
    
    // Allocate host memory for vectors
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // Verify memory allocation
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize host input vectors with random data
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory for vectors
    float *d_A = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_A, size));
    
    float *d_B = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_B, size));
    
    float *d_C = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_C, size));
    
    // Copy host vectors to device
    printf("Copying input data from the host memory to the CUDA device\n");
    checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Launch the Vector Addition CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // First perform vector addition
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the device result vector back to host
    printf("Copying output data from CUDA device to the host memory\n");
    checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Verify the addition result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    printf("Vector addition test PASSED\n");
    
    // Now perform element-wise multiplication (common in ML)
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorMultiply kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the result back to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Verify the multiplication result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] * h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Multiplication verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    printf("Vector multiplication test PASSED\n");
    
    // Apply ReLU activation function (simulate neural network layer)
    // First, set some negative values in d_C for demonstration
    for (int i = 0; i < numElements; ++i) {
        h_C[i] = -1.0f * rand() / (float)RAND_MAX; // Negative values
        if (i % 2 == 0) {
            h_C[i] *= -1.0f; // Make some values positive
        }
    }
    
    // Copy to device
    checkCudaErrors(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));
    
    // Apply ReLU
    reluActivation<<<blocksPerGrid, threadsPerBlock>>>(d_C, numElements);
    err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch ReLU kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy result back
    checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Verify ReLU (all values should be >= 0)
    bool reluCorrect = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_C[i] < 0) {
            fprintf(stderr, "ReLU verification failed at element %d: %f\n", i, h_C[i]);
            reluCorrect = false;
            break;
        }
    }
    
    if (reluCorrect) {
        printf("ReLU activation test PASSED\n");
    }
    
    // Free device global memory
    err = cudaFree(d_A);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(d_B);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(d_C);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    // Reset the device and exit
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    printf("Done\n");
    return 0;
}