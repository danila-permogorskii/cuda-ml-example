#include "../include/neural_network.h"
#include "../include/neural_network_kernels.cuh"
#include <math.h>
#include <curand_kernel.h>

// CUDA kernel for matrix multiplication: C = A * B
// A: m×k matrix, B: k×n matrix, C: m×n matrix
__global__ void matrixMultiply(const float* A, const float* B, float* C, 
                              int m, int n, int k) {
    // Calculate global row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within matrix bounds
    if (row < m && col < n) {
        float sum = 0.0f;
        
        // Compute dot product of row of A and column of B
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        
        // Store the result in C
        C[row * n + col] = sum;
    }
}

// CUDA kernel for matrix-vector addition: A = A + b
// A: m×n matrix, b: vector of length n (added to each row)
__global__ void addBiasToMatrix(float* A, const float* bias, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        // Add bias to each element in its column
        A[row * n + col] += bias[col];
    }
}

// CUDA kernel for ReLU activation function: A = max(0, A)
__global__ void reluActivation(float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // ReLU: max(0, x)
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}

// CUDA kernel for ReLU derivative
__global__ void reluDerivative(const float* A, float* dA, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Derivative of ReLU: 1 if x > 0, 0 otherwise
        dA[idx] = (A[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// CUDA kernel for matrix transpose: B = A^T
__global__ void matrixTranspose(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

// CUDA kernel for element-wise matrix multiplication: C = A * B
__global__ void elementWiseMultiply(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// CUDA kernel to update weights with gradient descent: W = W - learning_rate * dW
__global__ void updateWeights(float* W, const float* dW, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        W[idx] -= learning_rate * dW[idx];
    }
}

// CUDA kernel for weight initialization with Xavier/Glorot method
__global__ void initializeWeights(float* W, int fan_in, int fan_out, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = fan_in * fan_out;
    
    if (idx < size) {
        // Setup random state
        curandState_t state;
        curand_init(seed, idx, 0, &state);
        
        // Xavier/Glorot initialization: uniform distribution between -sqrt(6/(fan_in+fan_out)) and sqrt(6/(fan_in+fan_out))
        float limit = sqrtf(6.0f / (fan_in + fan_out));
        W[idx] = -limit + 2.0f * limit * curand_uniform(&state);
    }
}

// CUDA kernel for computing mean squared error loss
__global__ void computeMSELoss(const float* predictions, const float* targets, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        loss[idx] = diff * diff;
    }
}

// CUDA kernel for computing softmax activation
__global__ void softmaxActivation(float* input, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Find maximum value for numerical stability
        float max_val = input[idx * num_classes];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[idx * num_classes + i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            input[idx * num_classes + i] = expf(input[idx * num_classes + i] - max_val);
            sum += input[idx * num_classes + i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            input[idx * num_classes + i] /= sum;
        }
    }
}
