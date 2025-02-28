#include "../include/neural_network.h"
#include <math.h>
#include <time.h>

// Function declarations for CUDA kernels (defined in neural_network_kernels.cu)
extern "C" {
    __global__ void matrixMultiply(const float* A, const float* B, float* C, int m, int n, int k);
    __global__ void addBiasToMatrix(float* A, const float* bias, int m, int n);
    __global__ void reluActivation(float* A, int size);
    __global__ void reluDerivative(const float* A, float* dA, int size);
    __global__ void matrixTranspose(const float* A, float* B, int rows, int cols);
    __global__ void elementWiseMultiply(const float* A, const float* B, float* C, int size);
    __global__ void updateWeights(float* W, const float* dW, float learning_rate, int size);
    __global__ void initializeWeights(float* W, int fan_in, int fan_out, unsigned long seed);
    __global__ void computeMSELoss(const float* predictions, const float* targets, float* loss, int size);
    __global__ void softmaxActivation(float* input, int batch_size, int num_classes);
}

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
    
    // Initialize gradients to zero
    CHECK_CUDA_ERROR(cudaMemset(nn->d_dW1, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(nn->d_db1, 0, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(nn->d_dW2, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(nn->d_db2, 0, OUTPUT_SIZE * sizeof(float)));
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
    // You could add softmax here if needed:
    // softmaxActivation<<<(BATCH_SIZE + 255) / 256, 256>>>(nn->d_z2, BATCH_SIZE, OUTPUT_SIZE);
    
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

// Train neural network for a given number of epochs
void trainNeuralNetwork(NeuralNetwork* nn, const float* d_input, const float* d_target, int epochs) {
    printf("Training neural network with %d input features, %d hidden neurons, and %d outputs\n", 
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Batch size: %d, Learning rate: %f\n", BATCH_SIZE, LEARNING_RATE);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        forwardPass(nn, d_input);
        
        // Backward pass (with gradient descent update)
        backwardPass(nn, d_input, d_target);
        
        // Calculate and report loss (in a real implementation)
        // Here we would normally compute the loss and report it
        
        printf("Completed epoch %d\n", epoch + 1);
    }
    
    printf("Neural network training completed\n");
}

// Calculate loss (mean squared error)
float calculateLoss(const float* predictions, const float* targets, int size) {
    // Allocate device memory for loss
    float* d_loss;
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, size * sizeof(float)));
    
    // Compute mean squared error
    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    computeMSELoss<<<gridDim, blockDim>>>(predictions, targets, d_loss, size);
    
    // Copy loss to host
    float* h_loss = (float*)malloc(size * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(h_loss, d_loss, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate mean loss
    float total_loss = 0.0f;
    for (int i = 0; i < size; i++) {
        total_loss += h_loss[i];
    }
    
    // Free memory
    cudaFree(d_loss);
    free(h_loss);
    
    return total_loss / size;
}

// Predict for a single sample
void predictSingleSample(NeuralNetwork* nn, const float* sample, float* output) {
    // Copy sample to device
    float* d_sample;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sample, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sample, sample, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Forward pass (modified for single sample)
    // Define block and grid dimensions for matrix operations
    dim3 blockDim(16, 16);
    dim3 gridDim1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                 (1 + blockDim.y - 1) / blockDim.y); // 1 sample
    dim3 gridDim2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                 (1 + blockDim.y - 1) / blockDim.y); // 1 sample
    
    // Allocate temporary memory for this prediction
    float* d_z1_temp;
    float* d_a1_temp;
    float* d_z2_temp;
    float* d_output;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_z1_temp, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_a1_temp, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_z2_temp, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));
    
    // Layer 1: input -> hidden
    matrixMultiply<<<gridDim1, blockDim>>>(d_sample, nn->d_W1, d_z1_temp, 
                                          1, HIDDEN_SIZE, INPUT_SIZE);
    
    addBiasToMatrix<<<gridDim1, blockDim>>>(d_z1_temp, nn->d_b1, 1, HIDDEN_SIZE);
    
    dim3 blockDim1D(256);
    dim3 gridDim1D((HIDDEN_SIZE + blockDim1D.x - 1) / blockDim1D.x);
    reluActivation<<<gridDim1D, blockDim1D>>>(d_z1_temp, HIDDEN_SIZE);
    
    // Copy to a1_temp
    cudaMemcpy(d_a1_temp, d_z1_temp, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Layer 2: hidden -> output
    matrixMultiply<<<gridDim2, blockDim>>>(d_a1_temp, nn->d_W2, d_z2_temp, 
                                          1, OUTPUT_SIZE, HIDDEN_SIZE);
    
    addBiasToMatrix<<<gridDim2, blockDim>>>(d_z2_temp, nn->d_b2, 1, OUTPUT_SIZE);
    
    // Copy result to output
    cudaMemcpy(d_output, d_z2_temp, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Apply softmax for classification (optional)
    // softmaxActivation<<<1, 256>>>(d_output, 1, OUTPUT_SIZE);
    
    // Copy result back to host
    cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free temporary memory
    cudaFree(d_sample);
    cudaFree(d_z1_temp);
    cudaFree(d_a1_temp);
    cudaFree(d_z2_temp);
    cudaFree(d_output);
}
