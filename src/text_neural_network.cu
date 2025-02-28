#include "../include/text_neural_network.h"
#include "../include/text_neural_network_kernels.cuh"
#include <ctime>
#include <random>
#include <iostream>

// Function to initialize the text neural network
void initializeTextNeuralNetwork(TextNeuralNetwork* nn) {
    std::cout << "Initializing text neural network..." << std::endl;
    
    // Initialize vocabulary maps
    nn->vocabSize = 0;
    nn->wordToIndex.clear();
    nn->indexToWord.clear();
    
    // Add special tokens
    nn->wordToIndex[PAD_TOKEN] = nn->vocabSize;
    nn->indexToWord.push_back(PAD_TOKEN);
    nn->vocabSize++;
    
    nn->wordToIndex[UNK_TOKEN] = nn->vocabSize;
    nn->indexToWord.push_back(UNK_TOKEN);
    nn->vocabSize++;
    
    std::cout << "Added special tokens: " << PAD_TOKEN << ", " << UNK_TOKEN << std::endl;
    
    // Allocate GPU memory for embeddings and network parameters
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_embeddings, VOCABULARY_SIZE * EMBEDDING_DIM * sizeof(float)));
    
    // Allocate memory for weights and biases
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_W1, EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_W2, HIDDEN_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_b2, NUM_CLASSES * sizeof(float)));
    
    // Allocate memory for activations
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_text_indices, BATCH_SIZE * MAX_SEQUENCE_LENGTH * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_word_embeddings, BATCH_SIZE * MAX_SEQUENCE_LENGTH * EMBEDDING_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_text_embedding, BATCH_SIZE * EMBEDDING_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_z1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_a1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_z2, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_a2, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    
    // Allocate memory for gradients
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_da2, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dz2, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dW2, HIDDEN_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_db2, NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_da1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dz1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dW1, EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_db1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dtext_embedding, BATCH_SIZE * EMBEDDING_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&nn->d_dembeddings, VOCABULARY_SIZE * EMBEDDING_DIM * sizeof(float)));
    
    std::cout << "Allocated GPU memory for network parameters" << std::endl;
    
    // Initialize embeddings
    dim3 embBlockDim(EMBEDDING_DIM);
    dim3 embGridDim(VOCABULARY_SIZE);
    initializeEmbeddings<<<embGridDim, embBlockDim>>>(nn->d_embeddings, VOCABULARY_SIZE, EMBEDDING_DIM, time(NULL));
    
    // Initialize weights with small random values using Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Use Xavier/Glorot initialization for weights
    // This scales the random values based on the dimensions of the matrices
    float w1_scale = sqrtf(6.0f / (EMBEDDING_DIM + HIDDEN_SIZE));
    float w2_scale = sqrtf(6.0f / (HIDDEN_SIZE + NUM_CLASSES));
    
    std::uniform_real_distribution<float> w1_dist(-w1_scale, w1_scale);
    std::uniform_real_distribution<float> w2_dist(-w2_scale, w2_scale);
    
    // Allocate host memory for weights and biases
    float* h_W1 = new float[EMBEDDING_DIM * HIDDEN_SIZE];
    float* h_W2 = new float[HIDDEN_SIZE * NUM_CLASSES];
    float* h_b1 = new float[HIDDEN_SIZE];
    float* h_b2 = new float[NUM_CLASSES];
    
    // Initialize weights
    for (int i = 0; i < EMBEDDING_DIM * HIDDEN_SIZE; i++) {
        h_W1[i] = w1_dist(gen);
    }
    for (int i = 0; i < HIDDEN_SIZE * NUM_CLASSES; i++) {
        h_W2[i] = w2_dist(gen);
    }
    
    // Initialize biases to zero
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_b1[i] = 0.0f;
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        h_b2[i] = 0.0f;
    }
    
    // Copy weights and biases to device
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_W1, h_W1, EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_W2, h_W2, HIDDEN_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_b1, h_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_b2, h_b2, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free host memory
    delete[] h_W1;
    delete[] h_W2;
    delete[] h_b1;
    delete[] h_b2;
    
    // Initialize gradients to zero
    CHECK_CUDA_ERROR(cudaMemset(nn->d_dW1, 0, EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(nn->d_db1, 0, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(nn->d_dW2, 0, HIDDEN_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(nn->d_db2, 0, NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(nn->d_dembeddings, 0, VOCABULARY_SIZE * EMBEDDING_DIM * sizeof(float)));
    
    std::cout << "Initialized network parameters" << std::endl;
    std::cout << "Network architecture: " << INPUT_SIZE << " -> " << HIDDEN_SIZE 
              << " -> " << NUM_CLASSES << std::endl;
    std::cout << "Vocabulary size: " << VOCABULARY_SIZE << ", Embedding dimension: " 
              << EMBEDDING_DIM << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << ", Sequence length: " 
              << MAX_SEQUENCE_LENGTH << std::endl;
}

// Function to free the text neural network
void freeTextNeuralNetwork(TextNeuralNetwork* nn) {
    std::cout << "Freeing text neural network resources..." << std::endl;
    
    // Free device memory
    if (nn->d_embeddings) cudaFree(nn->d_embeddings);
    if (nn->d_W1) cudaFree(nn->d_W1);
    if (nn->d_b1) cudaFree(nn->d_b1);
    if (nn->d_W2) cudaFree(nn->d_W2);
    if (nn->d_b2) cudaFree(nn->d_b2);
    
    if (nn->d_text_indices) cudaFree(nn->d_text_indices);
    if (nn->d_word_embeddings) cudaFree(nn->d_word_embeddings);
    if (nn->d_text_embedding) cudaFree(nn->d_text_embedding);
    if (nn->d_z1) cudaFree(nn->d_z1);
    if (nn->d_a1) cudaFree(nn->d_a1);
    if (nn->d_z2) cudaFree(nn->d_z2);
    if (nn->d_a2) cudaFree(nn->d_a2);
    
    if (nn->d_da2) cudaFree(nn->d_da2);
    if (nn->d_dz2) cudaFree(nn->d_dz2);
    if (nn->d_dW2) cudaFree(nn->d_dW2);
    if (nn->d_db2) cudaFree(nn->d_db2);
    if (nn->d_da1) cudaFree(nn->d_da1);
    if (nn->d_dz1) cudaFree(nn->d_dz1);
    if (nn->d_dW1) cudaFree(nn->d_dW1);
    if (nn->d_db1) cudaFree(nn->d_db1);
    if (nn->d_dtext_embedding) cudaFree(nn->d_dtext_embedding);
    if (nn->d_dembeddings) cudaFree(nn->d_dembeddings);
    
    // Clear vocabulary
    nn->wordToIndex.clear();
    nn->indexToWord.clear();
    nn->vocabSize = 0;
    
    std::cout << "Neural network resources freed" << std::endl;
}
