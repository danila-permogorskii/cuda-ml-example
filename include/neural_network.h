#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Constants for network configuration
#define INPUT_SIZE 784   // 28x28 input (e.g., MNIST)
#define HIDDEN_SIZE 128  // Hidden layer neurons
#define OUTPUT_SIZE 10   // 10 classes (e.g., digits 0-9)
#define BATCH_SIZE 64    // Number of samples processed at once
#define LEARNING_RATE 0.01f

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

// Function declarations
void allocateNeuralNetwork(NeuralNetwork* nn);
void freeNeuralNetwork(NeuralNetwork* nn);
void forwardPass(NeuralNetwork* nn, const float* d_input);
void backwardPass(NeuralNetwork* nn, const float* d_input, const float* d_target);
void trainNeuralNetwork(NeuralNetwork* nn, const float* d_input, const float* d_target, int epochs);
float calculateLoss(const float* predictions, const float* targets, int size);
void predictSingleSample(NeuralNetwork* nn, const float* sample, float* output);

#endif // NEURAL_NETWORK_H
