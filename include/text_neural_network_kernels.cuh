#ifndef TEXT_NEURAL_NETWORK_KERNELS_CUH
#define TEXT_NEURAL_NETWORK_KERNELS_CUH

#include "text_neural_network.h"

/**
 * CUDA kernel for embedding lookup
 * Retrieves embeddings for each word index in a batch
 * 
 * @param wordIndices Word indices [batchSize x sequenceLength]
 * @param embeddings Embedding matrix [vocabSize x embeddingDim]
 * @param output Output tensor for word embeddings [batchSize x sequenceLength x embeddingDim]
 * @param batchSize Number of samples in batch
 * @param sequenceLength Maximum sequence length
 * @param embeddingDim Dimension of embedding vectors
 * @param vocabSize Size of vocabulary
 */
__global__ void lookupEmbeddings(const int* wordIndices, const float* embeddings, 
                               float* output, int batchSize, int sequenceLength, 
                               int embeddingDim, int vocabSize);

/**
 * CUDA kernel to average word embeddings into a single text embedding
 * 
 * @param wordEmbeddings Input word embeddings [batchSize x sequenceLength x embeddingDim]
 * @param textEmbedding Output text embedding [batchSize x embeddingDim]
 * @param batchSize Number of samples in batch
 * @param sequenceLength Maximum sequence length
 * @param embeddingDim Dimension of embedding vectors
 */
__global__ void averageEmbeddings(const float* wordEmbeddings, float* textEmbedding,
                                int batchSize, int sequenceLength, int embeddingDim);

/**
 * CUDA kernel for matrix multiplication: C = A * B
 * 
 * @param A Input matrix A [m x k]
 * @param B Input matrix B [k x n]
 * @param C Output matrix C [m x n]
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
__global__ void matrixMultiply(const float* A, const float* B, float* C, 
                              int m, int n, int k);

/**
 * CUDA kernel for matrix-vector addition: A = A + b
 * 
 * @param A Input/output matrix A [m x n]
 * @param bias Bias vector b [n]
 * @param m Number of rows in A
 * @param n Number of columns in A
 */
__global__ void addBiasToMatrix(float* A, const float* bias, int m, int n);

/**
 * CUDA kernel for ReLU activation function: A = max(0, A)
 * 
 * @param A Input/output tensor [size]
 * @param size Number of elements in A
 */
__global__ void reluActivation(float* A, int size);

/**
 * CUDA kernel for ReLU derivative
 * 
 * @param A Input tensor [size]
 * @param dA Output derivative tensor [size]
 * @param size Number of elements in A and dA
 */
__global__ void reluDerivative(const float* A, float* dA, int size);

/**
 * CUDA kernel for softmax activation
 * 
 * @param input Input/output tensor [batchSize x numClasses]
 * @param batchSize Number of samples in batch
 * @param numClasses Number of classes
 */
__global__ void softmaxActivation(float* input, int batchSize, int numClasses);

/**
 * CUDA kernel for cross-entropy loss derivative
 * 
 * @param predictions Model predictions [batchSize x numClasses]
 * @param targets Target values [batchSize x numClasses]
 * @param gradient Output gradient [batchSize x numClasses]
 * @param batchSize Number of samples in batch
 * @param numClasses Number of classes
 */
__global__ void crossEntropyDerivative(const float* predictions, const float* targets, 
                                     float* gradient, int batchSize, int numClasses);

/**
 * CUDA kernel for matrix transpose: B = A^T
 * 
 * @param A Input matrix A [rows x cols]
 * @param B Output matrix B [cols x rows]
 * @param rows Number of rows in A
 * @param cols Number of columns in A
 */
__global__ void matrixTranspose(const float* A, float* B, int rows, int cols);

/**
 * CUDA kernel for element-wise matrix multiplication: C = A * B
 * 
 * @param A Input matrix A [size]
 * @param B Input matrix B [size]
 * @param C Output matrix C [size]
 * @param size Number of elements in A, B, and C
 */
__global__ void elementWiseMultiply(const float* A, const float* B, float* C, int size);

/**
 * CUDA kernel to update weights with gradient descent: W = W - learning_rate * dW
 * 
 * @param W Input/output weights [size]
 * @param dW Weight gradients [size]
 * @param learningRate Learning rate
 * @param size Number of elements in W and dW
 */
__global__ void updateWeights(float* W, const float* dW, float learningRate, int size);

/**
 * CUDA kernel to update embeddings for only words that appeared in the batch
 * 
 * @param embeddings Input/output embedding matrix [vocabSize x embeddingDim]
 * @param gradient Embedding gradients [batchSize x embeddingDim]
 * @param wordIndices Word indices [batchSize x sequenceLength]
 * @param learningRate Learning rate
 * @param batchSize Number of samples in batch
 * @param sequenceLength Maximum sequence length
 * @param embeddingDim Dimension of embedding vectors
 * @param vocabSize Size of vocabulary
 */
__global__ void updateEmbeddings(float* embeddings, const float* gradient, 
                               const int* wordIndices, float learningRate,
                               int batchSize, int sequenceLength, int embeddingDim,
                               int vocabSize);

/**
 * CUDA kernel for embedding initialization
 * 
 * @param embeddings Output embedding matrix [vocabSize x embeddingDim]
 * @param vocabSize Size of vocabulary
 * @param embeddingDim Dimension of embedding vectors
 * @param seed Random seed
 */
__global__ void initializeEmbeddings(float* embeddings, int vocabSize, 
                                   int embeddingDim, unsigned long seed);

#endif // TEXT_NEURAL_NETWORK_KERNELS_CUH
