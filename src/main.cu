#include "../include/neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to print array contents (for debugging)
void printArray(const char* name, const float* array, int size) {
    printf("%s: [", name);
    for (int i = 0; i < size; i++) {
        printf("%.4f", array[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

// Function to generate random data for training
void generateRandomData(float** input, float** target, int batch_size, int input_size, int output_size) {
    // Allocate host memory
    *input = (float*)malloc(batch_size * input_size * sizeof(float));
    *target = (float*)malloc(batch_size * output_size * sizeof(float));
    
    // Seed random number generator
    srand(time(NULL));
    
    // Generate random input data (between 0 and 1)
    for (int i = 0; i < batch_size * input_size; i++) {
        (*input)[i] = (float)rand() / RAND_MAX;
    }
    
    // Generate random target data
    // We'll create a simple pattern: if sum of input values > 0.5*input_size, activate first output
    for (int i = 0; i < batch_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += (*input)[i * input_size + j];
        }
        
        // Determine target class based on sum
        int target_class = (sum > 0.5 * input_size) ? 0 : 1;
        
        // One-hot encode the target
        for (int j = 0; j < output_size; j++) {
            (*target)[i * output_size + j] = (j == target_class) ? 1.0f : 0.0f;
        }
    }
}

// Function to train and test the neural network
void runNeuralNetworkDemo() {
    printf("Neural Network CUDA Example\n");
    printf("===========================\n");
    
    // Create and initialize the neural network
    NeuralNetwork nn;
    allocateNeuralNetwork(&nn);
    
    printf("Neural network architecture: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Memory allocated for neural network parameters\n");
    
    // Generate random training data
    float* h_input;
    float* h_target;
    generateRandomData(&h_input, &h_target, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE);
    
    printf("Generated random training data with %d samples\n", BATCH_SIZE);
    
    // Allocate GPU memory for input and target
    float* d_input;
    float* d_target;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Copy training data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_target, h_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Train the neural network
    int epochs = 20;
    printf("\nTraining neural network for %d epochs...\n", epochs);
    trainNeuralNetwork(&nn, d_input, d_target, epochs);
    
    // Test the neural network on a few samples
    printf("\nTesting neural network on 5 random samples:\n");
    
    // Create some test samples
    float* test_input = (float*)malloc(INPUT_SIZE * sizeof(float));
    float* test_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    for (int i = 0; i < 5; i++) {
        // Generate a random test sample
        float sum = 0.0f;
        for (int j = 0; j < INPUT_SIZE; j++) {
            test_input[j] = (float)rand() / RAND_MAX;
            sum += test_input[j];
        }
        
        // Determine expected class based on sum
        int expected_class = (sum > 0.5 * INPUT_SIZE) ? 0 : 1;
        
        // Predict using the trained neural network
        predictSingleSample(&nn, test_input, test_output);
        
        // Find the predicted class (index of highest value)
        int predicted_class = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (test_output[j] > test_output[predicted_class]) {
                predicted_class = j;
            }
        }
        
        // Print prediction results
        printf("Sample %d: ", i+1);
        printf("Sum = %.2f, ", sum);
        printf("Expected class = %d, ", expected_class);
        printf("Predicted outputs = [");
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            printf("%.4f", test_output[j]);
            if (j < OUTPUT_SIZE - 1) printf(", ");
        }
        printf("], Predicted class = %d ", predicted_class);
        
        if (predicted_class == expected_class) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
    }
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_target);
    freeNeuralNetwork(&nn);
    
    free(h_input);
    free(h_target);
    free(test_input);
    free(test_output);
    
    printf("\nNeural network demonstration completed\n");
}

int main() {
    runNeuralNetworkDemo();
    return 0;
}
