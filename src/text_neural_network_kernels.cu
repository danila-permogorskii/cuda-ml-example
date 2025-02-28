#include "../include/text_neural_network.h"
#include "../include/text_neural_network_kernels.cuh"
#include <math.h>
#include <curand_kernel.h>

// CUDA kernel for embedding lookup
__global__ void lookupEmbeddings(const int* wordIndices, const float* embeddings, 
                               float* output, int batchSize, int sequenceLength, 
                               int embeddingDim, int vocabSize) {
    // Each thread handles one embedding dimension for one word in the batch
    int batchIdx = blockIdx.x;    // Which sample in the batch
    int seqIdx = blockIdx.y;      // Which position in the sequence
    int embIdx = threadIdx.x;     // Which dimension of the embedding
    
    if (batchIdx < batchSize && seqIdx < sequenceLength && embIdx < embeddingDim) {
        // Get the word index for this position
        int wordIdx = wordIndices[batchIdx * sequenceLength + seqIdx];
        
        // Ensure the index is within vocabulary bounds
        if (wordIdx >= 0 && wordIdx < vocabSize) {
            // Copy the embedding value from embeddings[wordIdx, embIdx] to output[batchIdx, seqIdx, embIdx]
            output[(batchIdx * sequenceLength + seqIdx) * embeddingDim + embIdx] = 
                embeddings[wordIdx * embeddingDim + embIdx];
        } else {
            // If out of bounds, set to zero
            output[(batchIdx * sequenceLength + seqIdx) * embeddingDim + embIdx] = 0.0f;
        }
    }
}

// CUDA kernel to average word embeddings into a single text embedding
__global__ void averageEmbeddings(const float* wordEmbeddings, float* textEmbedding,
                                int batchSize, int sequenceLength, int embeddingDim) {
    // Each thread handles one embedding dimension for one batch item
    int batchIdx = blockIdx.x;    // Which sample in the batch
    int embIdx = threadIdx.x;     // Which dimension of the embedding
    
    if (batchIdx < batchSize && embIdx < embeddingDim) {
        float sum = 0.0f;
        int count = 0;
        
        // Sum up embeddings for all words in the sequence
        for (int seqIdx = 0; seqIdx < sequenceLength; seqIdx++) {
            // Get the embedding value at [batchIdx, seqIdx, embIdx]
            float val = wordEmbeddings[(batchIdx * sequenceLength + seqIdx) * embeddingDim + embIdx];
            
            // Count non-zero values (actual words, not padding)
            if (val != 0.0f) {
                sum += val;
                count++;
            }
        }
        
        // Average the sum (avoid division by zero)
        textEmbedding[batchIdx * embeddingDim + embIdx] = 
            (count > 0) ? (sum / count) : 0.0f;
    }
}

// CUDA kernel for matrix multiplication: C = A * B
__global__ void matrixMultiply(const float* A, const float* B, float* C, 
                              int m, int n, int k) {
    // Each thread computes one element of C
    // A is m×k, B is k×n, C is m×n
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row in A and C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column in B and C
    
    if (row < m && col < n) {
        float sum = 0.0f;
        
        // Compute dot product of row of A and column of B
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        
        // Store result in C
        C[row * n + col] = sum;
    }
}

// CUDA kernel for matrix-vector addition: A = A + b
__global__ void addBiasToMatrix(float* A, const float* bias, int m, int n) {
    // Each thread handles one element of the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row in A
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column in A
    
    if (row < m && col < n) {
        // Add bias[col] to A[row, col]
        A[row * n + col] += bias[col];
    }
}

// CUDA kernel for ReLU activation function: A = max(0, A)
__global__ void reluActivation(float* A, int size) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // ReLU activation: max(0, x)
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}

// CUDA kernel for ReLU derivative
__global__ void reluDerivative(const float* A, float* dA, int size) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Derivative of ReLU: 1 if x > 0, 0 otherwise
        dA[idx] = (A[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// CUDA kernel for softmax activation
__global__ void softmaxActivation(float* input, int batchSize, int numClasses) {
    // Each block handles one sample in the batch
    int batchIdx = blockIdx.x;
    
    if (batchIdx < batchSize) {
        // Find maximum value for numerical stability
        float maxVal = input[batchIdx * numClasses];
        for (int i = 1; i < numClasses; i++) {
            maxVal = fmaxf(maxVal, input[batchIdx * numClasses + i]);
        }
        
        // Compute exponentials and their sum
        float sum = 0.0f;
        for (int i = 0; i < numClasses; i++) {
            // Subtract max for numerical stability before exp
            float expVal = expf(input[batchIdx * numClasses + i] - maxVal);
            input[batchIdx * numClasses + i] = expVal;
            sum += expVal;
        }
        
        // Normalize to get probabilities
        for (int i = 0; i < numClasses; i++) {
            input[batchIdx * numClasses + i] /= sum;
        }
    }
}

// CUDA kernel for cross-entropy loss derivative
__global__ void crossEntropyDerivative(const float* predictions, const float* targets, 
                                     float* gradient, int batchSize, int numClasses) {
    // Each thread handles one class for one sample
    int batchIdx = blockIdx.x;    // Which sample in the batch
    int classIdx = threadIdx.x;   // Which class
    
    if (batchIdx < batchSize && classIdx < numClasses) {
        // Get index in the flattened arrays
        int idx = batchIdx * numClasses + classIdx;
        
        // Gradient of cross-entropy is (prediction - target)
        gradient[idx] = predictions[idx] - targets[idx];
    }
}

// CUDA kernel for matrix transpose: B = A^T
__global__ void matrixTranspose(const float* A, float* B, int rows, int cols) {
    // Each thread handles one element
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row in A
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column in A
    
    if (row < rows && col < cols) {
        // B[col, row] = A[row, col]
        B[col * rows + row] = A[row * cols + col];
    }
}

// CUDA kernel for element-wise matrix multiplication: C = A * B
__global__ void elementWiseMultiply(const float* A, const float* B, float* C, int size) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Element-wise multiplication
        C[idx] = A[idx] * B[idx];
    }
}

// CUDA kernel to update weights with gradient descent: W = W - learning_rate * dW
__global__ void updateWeights(float* W, const float* dW, float learningRate, int size) {
    // Each thread handles one weight
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Update weight using gradient descent
        W[idx] -= learningRate * dW[idx];
    }
}

// CUDA kernel to update embeddings for only words that appeared in the batch
__global__ void updateEmbeddings(float* embeddings, const float* gradient, 
                               const int* wordIndices, float learningRate,
                               int batchSize, int sequenceLength, int embeddingDim,
                               int vocabSize) {
    // Each thread handles one embedding dimension for one word occurrence
    int batchIdx = blockIdx.x;    // Which sample in the batch
    int seqIdx = blockIdx.y;      // Which position in the sequence
    int embIdx = threadIdx.x;     // Which dimension of the embedding
    
    if (batchIdx < batchSize && seqIdx < sequenceLength && embIdx < embeddingDim) {
        // Get the word index for this position
        int wordIdx = wordIndices[batchIdx * sequenceLength + seqIdx];
        
        // Only update if the word is in vocabulary
        if (wordIdx >= 0 && wordIdx < vocabSize) {
            // Calculate embedding gradient for this word
            // We use the same gradient for all occurrences of a word in the batch
            float grad = gradient[batchIdx * embeddingDim + embIdx];
            
            // Atomic update to prevent race conditions when the same word appears multiple times
            atomicAdd(&embeddings[wordIdx * embeddingDim + embIdx], -learningRate * grad);
        }
    }
}

// CUDA kernel for embedding initialization
__global__ void initializeEmbeddings(float* embeddings, int vocabSize, 
                                   int embeddingDim, unsigned long seed) {
    // Each thread initializes one element of the embedding matrix
    int wordIdx = blockIdx.x;     // Word index
    int embIdx = threadIdx.x;     // Embedding dimension
    
    if (wordIdx < vocabSize && embIdx < embeddingDim) {
        // Setup curand state for random number generation
        curandState_t state;
        curand_init(seed, wordIdx * embeddingDim + embIdx, 0, &state);
        
        // Initialize with small random values from uniform distribution
        // We use Xavier/Glorot initialization which scales values based on the dimension
        float scale = sqrtf(6.0f / embeddingDim);
        embeddings[wordIdx * embeddingDim + embIdx] = scale * (2.0f * curand_uniform(&state) - 1.0f);
    }
}
