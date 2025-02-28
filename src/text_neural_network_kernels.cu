#include "../include/text_neural_network.h"
#include "../include/text_neural_network_kernels.cuh"
#include <math.h>
#include <curand_kernel.h>

// CUDA kernel for embedding lookup
__global__ void lookupEmbeddings(const int* wordIndices, const float* embeddings, 
                               float* output, int batchSize, int sequenceLength, 
                               int embeddingDim, int vocabSize) {
    // Each thread handles one word in the batch
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int embIdx = threadIdx.x;
    
    if (batchIdx < batchSize && seqIdx < sequenceLength && embIdx < embeddingDim) {
        // Get the word index for this position
        int wordIdx = wordIndices[batchIdx * sequenceLength + seqIdx];
        
        // Ensure the index is within vocabulary bounds
        if (wordIdx >= 0 && wordIdx < vocabSize) {
            // Copy the embedding value
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
    int batchIdx = blockIdx.x;
    int embIdx = threadIdx.x;
    
    if (batchIdx < batchSize && embIdx < embeddingDim) {
        float sum = 0.0f;
        int count = 0;
        
        // Sum up embeddings for all words in the sequence
        for (int seqIdx = 0; seqIdx < sequenceLength; seqIdx++) {
            float val = wordEmbeddings[(batchIdx * sequenceLength + seqIdx) * embeddingDim + embIdx];
            // Only count non-zero values (actual words, not padding)
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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CUDA kernel for matrix-vector addition: A = A + b
__global__ void addBiasToMatrix(float* A, const float* bias, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        A[row * n + col] += bias[col];
    }
}

// CUDA kernel for ReLU activation function: A = max(0, A)
__global__ void reluActivation(float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}

// CUDA kernel for ReLU derivative
__global__ void reluDerivative(const float* A, float* dA, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        dA[idx] = (A[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// CUDA kernel for softmax activation
__global__ void softmaxActivation(float* input, int batchSize, int numClasses) {
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
    int batchIdx = blockIdx.x;
    int classIdx = threadIdx.x;
    
    if (batchIdx < batchSize && classIdx < numClasses) {
        // Gradient of cross-entropy is (prediction - target)
        int idx = batchIdx * numClasses + classIdx;
        gradient[idx] = predictions[idx] - targets[idx];
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
__global__ void updateWeights(float* W, const float* dW, float learningRate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        W[idx] -= learningRate * dW[idx];
    }
}

// CUDA kernel to update embeddings for only words that appeared in the batch
__global__ void updateEmbeddings(float* embeddings, const float* gradient, 
                               const int* wordIndices, float learningRate,
                               int batchSize, int sequenceLength, int embeddingDim,
                               int vocabSize) {
    // Each thread handles one embedding dimension for one word occurrence
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int embIdx = threadIdx.x;
    
    if (batchIdx < batchSize && seqIdx < sequenceLength && embIdx < embeddingDim) {
        // Get the word index for this position
        int wordIdx = wordIndices[batchIdx * sequenceLength + seqIdx];
        
        // Only update if the word is in vocabulary
        if (wordIdx >= 0 && wordIdx < vocabSize) {
            // Calculate embedding gradient for this word
            float grad = gradient[(batchIdx * sequenceLength + seqIdx) * embeddingDim + embIdx];
            
            // Atomic update to prevent race conditions when the same word appears multiple times
            atomicAdd(&embeddings[wordIdx * embeddingDim + embIdx], -learningRate * grad);
        }
    }
}

// CUDA kernel for embedding initialization
__global__ void initializeEmbeddings(float* embeddings, int vocabSize, 
                                   int embeddingDim, unsigned long seed) {
    int wordIdx = blockIdx.x;
    int embIdx = threadIdx.x;
    
    if (wordIdx < vocabSize && embIdx < embeddingDim) {
        // Setup curand state for random number generation
        curandState_t state;
        curand_init(seed, wordIdx * embeddingDim + embIdx, 0, &state);
        
        // Initialize with small random values from uniform distribution
        float scale = sqrtf(6.0f / embeddingDim);
        embeddings[wordIdx * embeddingDim + embIdx] = scale * (2.0f * curand_uniform(&state) - 1.0f);
    }
}
