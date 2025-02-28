#ifndef TEXT_NEURAL_NETWORK_H
#define TEXT_NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Constants for text neural network configuration
#define VOCABULARY_SIZE 5000        // Maximum number of words in vocabulary
#define MAX_SEQUENCE_LENGTH 100     // Maximum number of words per text sample
#define EMBEDDING_DIM 64            // Size of word embedding vectors
#define HIDDEN_SIZE 128             // Size of hidden layer
#define NUM_CLASSES 2               // Number of output classes (positive/negative)
#define BATCH_SIZE 32               // Number of samples per batch
#define LEARNING_RATE 0.01f         // Learning rate for gradient descent

// Special tokens
#define UNK_TOKEN "<UNK>"    // Unknown word token
#define PAD_TOKEN "<PAD>"    // Padding token

// Structure for text neural network
struct TextNeuralNetwork {
    // Vocabulary
    std::unordered_map<std::string, int> wordToIndex;  // Maps words to indices
    std::vector<std::string> indexToWord;              // Maps indices to words
    int vocabSize;                                     // Actual vocabulary size

    // Word embeddings
    float* d_embeddings;     // Device memory for embedding matrix [vocabSize x embeddingDim]
    
    // Weights between text representation and hidden layer
    float* d_W1;             // Device memory for weights [embeddingDim x hiddenSize]
    float* d_b1;             // Device memory for biases [hiddenSize]
    
    // Weights between hidden layer and output
    float* d_W2;             // Device memory for weights [hiddenSize x numClasses]
    float* d_b2;             // Device memory for biases [numClasses]
    
    // Activations for the current batch
    float* d_text_indices;   // Device memory for word indices [batchSize x maxSequenceLength]
    float* d_word_embeddings;// Device memory for word embeddings [batchSize x maxSequenceLength x embeddingDim]
    float* d_text_embedding; // Device memory for averaged text embedding [batchSize x embeddingDim]
    float* d_z1;             // Device memory for hidden layer pre-activation [batchSize x hiddenSize]
    float* d_a1;             // Device memory for hidden layer activation [batchSize x hiddenSize]
    float* d_z2;             // Device memory for output layer pre-activation [batchSize x numClasses]
    float* d_a2;             // Device memory for output layer activation (predictions) [batchSize x numClasses]
    
    // Gradients for backpropagation
    float* d_da2;            // Device memory for output layer gradient [batchSize x numClasses]
    float* d_dz2;            // Device memory for output layer pre-activation gradient [batchSize x numClasses]
    float* d_dW2;            // Device memory for output weights gradient [hiddenSize x numClasses]
    float* d_db2;            // Device memory for output biases gradient [numClasses]
    float* d_da1;            // Device memory for hidden layer gradient [batchSize x hiddenSize]
    float* d_dz1;            // Device memory for hidden layer pre-activation gradient [batchSize x hiddenSize]
    float* d_dW1;            // Device memory for hidden weights gradient [embeddingDim x hiddenSize]
    float* d_db1;            // Device memory for hidden biases gradient [hiddenSize]
    float* d_dtext_embedding;// Device memory for text embedding gradient [batchSize x embeddingDim]
    float* d_dembeddings;    // Device memory for embedding matrix gradient [vocabSize x embeddingDim]
};

// Function declarations for text preprocessing
void buildVocabulary(TextNeuralNetwork* nn, const std::vector<std::string>& texts);
std::vector<std::string> tokenize(const std::string& text);
void preprocessBatch(TextNeuralNetwork* nn, const std::vector<std::string>& texts, int* indices);
void loadSentimentDataset(std::vector<std::string>& texts, std::vector<int>& labels, const std::string& filename);

// Function declarations for neural network operations
void initializeTextNeuralNetwork(TextNeuralNetwork* nn);
void freeTextNeuralNetwork(TextNeuralNetwork* nn);
void forwardPassText(TextNeuralNetwork* nn, int* textIndices);
void backwardPassText(TextNeuralNetwork* nn, int* textIndices, float* targets);
void trainTextNeuralNetwork(TextNeuralNetwork* nn, const std::vector<std::string>& texts, const std::vector<int>& labels, int epochs);
void predictSentiment(TextNeuralNetwork* nn, const std::string& text, float* output);
float calculateAccuracy(const float* predictions, const int* labels, int numSamples);

#endif // TEXT_NEURAL_NETWORK_H
