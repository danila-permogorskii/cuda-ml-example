#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Constants for network configuration
const int INPUT_SIZE = 784;   // 28x28 input (e.g., MNIST)
const int HIDDEN_SIZE = 128;  // Hidden layer neurons
const int OUTPUT_SIZE = 10;   // 10 classes (e.g., digits 0-9)
const int BATCH_SIZE = 64;    // Number of samples processed at once
const float LEARNING_RATE = 0.01f;

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

// Structure to hold neural network parameters
struct NeuralNetwork {
    // Weights and biases
    float *d_W1, *d_b1;  // Weights and bias for input -> hidden
    float *d_W2, *d_b2;  // Weights and bias for hidden -> output
    
    // Activations
    float *d_z1, *d_a1;  // Pre-activation and activation for hidden layer
    float *d_z2, *d_a2;  // Pre-activation and activation for output layer
    
    // Gradients
    float *d_dW1, *d_db1;  // Gradients for W1 and b1
    float *d_dW2, *d_db2;  // Gradients for W2 and b2
    float *d_dz1, *d_dz2;  // Gradients for pre-activations
    float *d_da1, *d_da2;  // Gradients for activations
    
    // Temporary buffers
    float *d_temp1, *d_temp2;
};

// Allocate memory for neural network parameters
void allocateNeuralNetwork(NeuralNetwork* nn) {
    // Allocate memory for weights and biases
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_b2, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate memory for activations
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_z1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_a1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_z2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_a2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate memory for gradients
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dW1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_db1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dW2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_db2, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dz1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dz2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_da1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_da2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate memory for temporary buffers
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_temp1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_temp2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Initialize weights and biases
    dim3 blockDim(256);
    dim3 gridDimW1((INPUT_SIZE * HIDDEN_SIZE + blockDim.x - 1) / blockDim.x);
    dim3 gridDimW2((HIDDEN_SIZE * OUTPUT_SIZE + blockDim.x - 1) / blockDim.x);
    
    // Initialize weights using Xavier/Glorot initialization
    initializeWeights<<<gridDimW1, blockDim>>>(nn->d_W1, INPUT_SIZE, HIDDEN_SIZE, time(NULL));
    initializeWeights<<<gridDimW2, blockDim>>>(nn->d_W2, HIDDEN_SIZE, OUTPUT_SIZE, time(NULL) + 1000);
    
    // Initialize biases to zero
    CHECK_CUDA_ERROR(cudaMemset(nn->d_b1, 0, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(nn->d_b2, 0, OUTPUT_SIZE * sizeof(float)));
}

// Free neural network memory
void freeNeuralNetwork(NeuralNetwork* nn) {
    // Free weights and biases
    cudaFree(nn->d_W1);
    cudaFree(nn->d_b1);
    cudaFree(nn->d_W2);
    cudaFree(nn->d_b2);
    
    // Free activations
    cudaFree(nn->d_z1);
    cudaFree(nn->d_a1);
    cudaFree(nn->d_z2);
    cudaFree(nn->d_a2);
    
    // Free gradients
    cudaFree(nn->d_dW1);
    cudaFree(nn->d_db1);
    cudaFree(nn->d_dW2);
    cudaFree(nn->d_db2);
    cudaFree(nn->d_dz1);
    cudaFree(nn->d_dz2);
    cudaFree(nn->d_da1);
    cudaFree(nn->d_da2);
    
    // Free temporary buffers
    cudaFree(nn->d_temp1);
    cudaFree(nn->d_temp2);
}

// Forward pass of neural network
void forwardPass(NeuralNetwork* nn, const float* d_input) {
    // Define block and grid dimensions for matrix operations
    dim3 blockDim(16, 16);
    dim3 gridDim1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                 (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
    dim3 gridDim2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                 (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
    
    // Layer 1: input -> hidden
    // Z1 = X * W1
    matrixMultiply<<<gridDim1, blockDim>>>(d_input, nn->d_W1, nn->d_z1, 
                                          BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
    
    // Z1 = Z1 + b1
    addBiasToMatrix<<<gridDim1, blockDim>>>(nn->d_z1, nn->d_b1, BATCH_SIZE, HIDDEN_SIZE);
    
    // A1 = ReLU(Z1)
    dim3 blockDim1D(256);
    dim3 gridDim1D((BATCH_SIZE * HIDDEN_SIZE + blockDim1D.x - 1) / blockDim1D.x);
    reluActivation<<<gridDim1D, blockDim1D>>>(nn->d_z1, BATCH_SIZE * HIDDEN_SIZE);
    
    // Copy Z1 to A1 (since we apply ReLU directly to Z1)
    cudaMemcpy(nn->d_a1, nn->d_z1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Layer 2: hidden -> output
    // Z2 = A1 * W2
    matrixMultiply<<<gridDim2, blockDim>>>(nn->d_a1, nn->d_W2, nn->d_z2, 
                                          BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
    
    // Z2 = Z2 + b2
    addBiasToMatrix<<<gridDim2, blockDim>>>(nn->d_z2, nn->d_b2, BATCH_SIZE, OUTPUT_SIZE);
    
    // For simplicity, no activation on output layer (would be softmax for classification)
    cudaMemcpy(nn->d_a2, nn->d_z2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Backward pass of neural network
void backwardPass(NeuralNetwork* nn, const float* d_input, const float* d_target) {
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                 (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
    dim3 gridDim2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                 (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
    dim3 gridDimW1((INPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                  (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    dim3 gridDimW2((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                  (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y);
    
    // Calculate output layer error (dA2 = A2 - target)
    // For mean squared error: dA2 = (A2 - target) / BATCH_SIZE
    dim3 blockDim1D(256);
    dim3 gridDim1D_output((BATCH_SIZE * OUTPUT_SIZE + blockDim1D.x - 1) / blockDim1D.x);
    dim3 gridDim1D_hidden((BATCH_SIZE * HIDDEN_SIZE + blockDim1D.x - 1) / blockDim1D.x);
    
    // For this simplified example, we'll just copy the output layer error directly
    // (assuming it's already calculated and passed in as d_target)
    cudaMemcpy(nn->d_da2, d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Calculate gradients for output layer weights
    // dW2 = A1^T * dA2
    float* d_a1_T;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a1_T, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    matrixTranspose<<<gridDim1, blockDim>>>(nn->d_a1, d_a1_T, BATCH_SIZE, HIDDEN_SIZE);
    
    matrixMultiply<<<gridDimW2, blockDim>>>(d_a1_T, nn->d_da2, nn->d_dW2,
                                           HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
    
    // Calculate gradients for hidden layer
    // dZ1 = (dA2 * W2^T) * ReLU'(Z1)
    float* d_W2_T;
    CHECK_CUDA_ERROR(cudaMalloc(&d_W2_T, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    matrixTranspose<<<gridDimW2, blockDim>>>(nn->d_W2, d_W2_T, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Calculate dA1 = dA2 * W2^T
    matrixMultiply<<<gridDim1, blockDim>>>(nn->d_da2, d_W2_T, nn->d_da1,
                                          BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Calculate ReLU derivative
    reluDerivative<<<gridDim1D_hidden, blockDim1D>>>(nn->d_z1, nn->d_dz1, BATCH_SIZE * HIDDEN_SIZE);
    
    // Calculate dZ1 = dA1 * ReLU'(Z1)
    elementWiseMultiply<<<gridDim1D_hidden, blockDim1D>>>(nn->d_da1, nn->d_dz1, nn->d_dz1, BATCH_SIZE * HIDDEN_SIZE);
    
    // Calculate gradients for input layer weights
    // dW1 = X^T * dZ1
    float* d_input_T;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_T, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    
    // Define grid dimensions for input matrix transpose
    dim3 gridDimInputT((INPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                      (BATCH_SIZE + blockDim.y - 1) / blockDim.y);
    
    // Transpose input matrix
    matrixTranspose<<<gridDimInputT, blockDim>>>(d_input, d_input_T, BATCH_SIZE, INPUT_SIZE);
    
    matrixMultiply<<<gridDimW1, blockDim>>>(d_input_T, nn->d_dz1, nn->d_dW1,
                                           INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE);
    
    // Free temporary memory
    cudaFree(d_a1_T);
    cudaFree(d_W2_T);
    cudaFree(d_input_T);
    
    // Update weights and biases
    dim3 gridDimW1_1D((INPUT_SIZE * HIDDEN_SIZE + blockDim1D.x - 1) / blockDim1D.x);
    dim3 gridDimW2_1D((HIDDEN_SIZE * OUTPUT_SIZE + blockDim1D.x - 1) / blockDim1D.x);
    dim3 gridDimB1((HIDDEN_SIZE + blockDim1D.x - 1) / blockDim1D.x);
    dim3 gridDimB2((OUTPUT_SIZE + blockDim1D.x - 1) / blockDim1D.x);
    
    updateWeights<<<gridDimW1_1D, blockDim1D>>>(nn->d_W1, nn->d_dW1, LEARNING_RATE, INPUT_SIZE * HIDDEN_SIZE);
    updateWeights<<<gridDimW2_1D, blockDim1D>>>(nn->d_W2, nn->d_dW2, LEARNING_RATE, HIDDEN_SIZE * OUTPUT_SIZE);
    updateWeights<<<gridDimB1, blockDim1D>>>(nn->d_b1, nn->d_db1, LEARNING_RATE, HIDDEN_SIZE);
    updateWeights<<<gridDimB2, blockDim1D>>>(nn->d_b2, nn->d_db2, LEARNING_RATE, OUTPUT_SIZE);
}

// Simple demonstration of neural network on random data
int main() {
    // Allocate neural network on GPU
    NeuralNetwork nn;
    allocateNeuralNetwork(&nn);
    
    // Generate random input data (normally would load from a dataset)
    float* h_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float* h_target = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    
    // Generate random input data (0 to 1)
    for (int i = 0; i < BATCH_SIZE * INPUT_SIZE; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    // Generate random target data (0 or 1 for simplicity)
    for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; i++) {
        h_target[i] = (rand() % 2) ? 1.0f : 0.0f;
    }
    
    // Allocate GPU memory for input and target
    float* d_input;
    float* d_target;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Copy input and target to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_target, h_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Training loop (just a few iterations for demonstration)
    printf("Training neural network with %d input features, %d hidden neurons, and %d outputs\n", 
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Batch size: %d, Learning rate: %f\n", BATCH_SIZE, LEARNING_RATE);
    
    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward pass
        forwardPass(&nn, d_input);
        
        // Backward pass (with gradient descent update)
        backwardPass(&nn, d_input, d_target);
        
        // In a real implementation, we'd calculate and report loss here
        printf("Completed epoch %d\n", epoch + 1);
    }
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_target);
    freeNeuralNetwork(&nn);
    
    // Free host memory
    free(h_input);
    free(h_target);
    
    printf("Neural network training completed\n");
    return 0;
}
