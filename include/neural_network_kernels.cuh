#ifndef NEURAL_NETWORK_KERNELS_CUH
#define NEURAL_NETWORK_KERNELS_CUH

// CUDA kernel declarations with the correct __global__ qualifier
// Note that the .cuh extension is important for CUDA headers

// CUDA kernel for matrix multiplication: C = A * B
// A: m×k matrix, B: k×n matrix, C: m×n matrix
__global__ void matrixMultiply(const float* A, const float* B, float* C, 
                              int m, int n, int k);

// CUDA kernel for matrix-vector addition: A = A + b
// A: m×n matrix, b: vector of length n (added to each row)
__global__ void addBiasToMatrix(float* A, const float* bias, int m, int n);

// CUDA kernel for ReLU activation function: A = max(0, A)
__global__ void reluActivation(float* A, int size);

// CUDA kernel for ReLU derivative
__global__ void reluDerivative(const float* A, float* dA, int size);

// CUDA kernel for matrix transpose: B = A^T
__global__ void matrixTranspose(const float* A, float* B, int rows, int cols);

// CUDA kernel for element-wise matrix multiplication: C = A * B
__global__ void elementWiseMultiply(const float* A, const float* B, float* C, int size);

// CUDA kernel to update weights with gradient descent: W = W - learning_rate * dW
__global__ void updateWeights(float* W, const float* dW, float learning_rate, int size);

// CUDA kernel for weight initialization with Xavier/Glorot method
__global__ void initializeWeights(float* W, int fan_in, int fan_out, unsigned long seed);

// CUDA kernel for computing mean squared error loss
__global__ void computeMSELoss(const float* predictions, const float* targets, float* loss, int size);

// CUDA kernel for computing softmax activation
__global__ void softmaxActivation(float* input, int batch_size, int num_classes);

#endif // NEURAL_NETWORK_KERNELS_CUH
