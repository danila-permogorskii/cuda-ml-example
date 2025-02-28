#ifndef TEXT_NEURAL_NETWORK_KERNELS_CUH
#define TEXT_NEURAL_NETWORK_KERNELS_CUH

#include "text_neural_network.h"

// CUDA kernel for embedding lookup
// Retrieves embeddings for each word index in a batch
__global__ void lookupEmbeddings(const int* wordIndices, const float* embeddings, 
                               float* output, int batchSize, int sequenceLength, 
                               int embeddingDim, int vocabSize);

// CUDA kernel to average word embeddings into a single text embedding
__global__ void averageEmbeddings(const float* wordEmbeddings, float* textEmbedding,
                                int batchSize, int sequenceLength, int embeddingDim);

// CUDA kernel for matrix multiplication: C = A * B
__global__ void matrixMultiply(const float* A, const float* B, float* C, 
                              int m, int n, int k);

// CUDA kernel for matrix-vector addition: A = A + b
__global__ void addBiasToMatrix(float* A, const float* bias, int m, int n);

// CUDA kernel for ReLU activation function: A = max(0, A)
__global__ void reluActivation(float* A, int size);

// CUDA kernel for ReLU derivative
__global__ void reluDerivative(const float* A, float* dA, int size);

// CUDA kernel for softmax activation
__global__ void softmaxActivation(float* input, int batchSize, int numClasses);

// CUDA kernel for cross-entropy loss derivative
__global__ void crossEntropyDerivative(const float* predictions, const float* targets, 
                                     float* gradient, int batchSize, int numClasses);

// CUDA kernel for matrix transpose: B = A^T
__global__ void matrixTranspose(const float* A, float* B, int rows, int cols);

// CUDA kernel for element-wise matrix multiplication: C = A * B
__global__ void elementWiseMultiply(const float* A, const float* B, float* C, int size);

// CUDA kernel to update weights with gradient descent: W = W - learning_rate * dW
__global__ void updateWeights(float* W, const float* dW, float learningRate, int size);

// CUDA kernel to update embeddings for only words that appeared in the batch
__global__ void updateEmbeddings(float* embeddings, const float* gradient, 
                               const int* wordIndices, float learningRate,
                               int batchSize, int sequenceLength, int embeddingDim,
                               int vocabSize);

// CUDA kernel for embedding initialization
__global__ void initializeEmbeddings(float* embeddings, int vocabSize, 
                                   int embeddingDim, unsigned long seed);

#endif // TEXT_NEURAL_NETWORK_KERNELS_CUH
