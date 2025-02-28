#include "../include/text_neural_network.h"
#include "../include/text_neural_network_kernels.cuh"
#include <iostream>
#include <algorithm>  // For std::random_shuffle
#include <cstring>   // For std::memset

// Backward pass for text neural network
void backwardPassText(TextNeuralNetwork* nn, int* textIndices, float* targets) {
    cudaError_t err;
    
    // ===============================================================
    // Step 1: Calculate output layer error (dA2 = A2 - targets)
    // ===============================================================
    
    // Copy targets to device
    float* d_targets;
    CHECK_CUDA_ERROR(cudaMalloc(&d_targets, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_targets, targets, 
                             BATCH_SIZE * NUM_CLASSES * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    // Calculate gradient of cross-entropy loss w.r.t. output layer activations
    dim3 outputBlock(NUM_CLASSES);
    dim3 outputGrid(BATCH_SIZE);
    
    crossEntropyDerivative<<<outputGrid, outputBlock>>>(
        nn->d_a2,              // Predicted probabilities [BATCH_SIZE x NUM_CLASSES]
        d_targets,             // Target values [BATCH_SIZE x NUM_CLASSES]
        nn->d_da2,             // Output gradient [BATCH_SIZE x NUM_CLASSES]
        BATCH_SIZE, NUM_CLASSES);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in crossEntropyDerivative kernel: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // ===============================================================
    // Step 2: Calculate gradients for output layer weights (W2)
    // ===============================================================
    
    // Transpose A1 for matrix multiplication
    float* d_a1_T;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a1_T, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    
    dim3 transposeBlock(16, 16);
    dim3 transposeGrid1(
        (HIDDEN_SIZE + transposeBlock.x - 1) / transposeBlock.x,
        (BATCH_SIZE + transposeBlock.y - 1) / transposeBlock.y);
    
    matrixTranspose<<<transposeGrid1, transposeBlock>>>(
        nn->d_a1,              // Input matrix [BATCH_SIZE x HIDDEN_SIZE]
        d_a1_T,                // Output transposed matrix [HIDDEN_SIZE x BATCH_SIZE]
        BATCH_SIZE, HIDDEN_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in A1 transpose kernel: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_targets);
        return;
    }
    
    // Calculate dW2 = A1^T * dA2
    dim3 denseBlock(16, 16);
    dim3 denseGridW2(
        (NUM_CLASSES + denseBlock.x - 1) / denseBlock.x,
        (HIDDEN_SIZE + denseBlock.y - 1) / denseBlock.y);
    
    // Zero out gradient buffer before accumulation
    CHECK_CUDA_ERROR(cudaMemset(nn->d_dW2, 0, HIDDEN_SIZE * NUM_CLASSES * sizeof(float)));
    
    matrixMultiply<<<denseGridW2, denseBlock>>>(
        d_a1_T,                // Transposed hidden activations [HIDDEN_SIZE x BATCH_SIZE]
        nn->d_da2,             // Output gradients [BATCH_SIZE x NUM_CLASSES]
        nn->d_dW2,             // Weight gradients [HIDDEN_SIZE x NUM_CLASSES]
        HIDDEN_SIZE, NUM_CLASSES, BATCH_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in dW2 calculation: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_targets);
        return;
    }
    
    // Calculate bias gradients (db2) - sum over batch dimension
    // This is a simplified approach where we set db2 = sum of dA2 across batch
    dim3 biasBlock(NUM_CLASSES);
    dim3 biasGrid(1);
    
    // Zero out bias gradient buffer
    CHECK_CUDA_ERROR(cudaMemset(nn->d_db2, 0, NUM_CLASSES * sizeof(float)));
    
    // Implement a custom kernel to sum gradients across batch dimension
    // For simplicity, we'll calculate this on the CPU
    float* h_da2 = new float[BATCH_SIZE * NUM_CLASSES];
    CHECK_CUDA_ERROR(cudaMemcpy(h_da2, nn->d_da2, 
                             BATCH_SIZE * NUM_CLASSES * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    
    float* h_db2 = new float[NUM_CLASSES]();
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            h_db2[j] += h_da2[b * NUM_CLASSES + j];
        }
    }
    
    // Copy back to device
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_db2, h_db2, 
                             NUM_CLASSES * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    delete[] h_da2;
    delete[] h_db2;
    
    // ===============================================================
    // Step 3: Calculate gradients for hidden layer
    // ===============================================================
    
    // Calculate dA1 = dA2 * W2^T
    // First, transpose W2
    float* d_W2_T;
    CHECK_CUDA_ERROR(cudaMalloc(&d_W2_T, HIDDEN_SIZE * NUM_CLASSES * sizeof(float)));
    
    dim3 transposeGrid2(
        (HIDDEN_SIZE + transposeBlock.x - 1) / transposeBlock.x,
        (NUM_CLASSES + transposeBlock.y - 1) / transposeBlock.y);
    
    matrixTranspose<<<transposeGrid2, transposeBlock>>>(
        nn->d_W2,              // Input matrix [HIDDEN_SIZE x NUM_CLASSES]
        d_W2_T,                // Output transposed matrix [NUM_CLASSES x HIDDEN_SIZE]
        HIDDEN_SIZE, NUM_CLASSES);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in W2 transpose kernel: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_targets);
        return;
    }
    
    // Calculate dA1 = dA2 * W2^T
    dim3 denseGrid1(
        (HIDDEN_SIZE + denseBlock.x - 1) / denseBlock.x,
        (BATCH_SIZE + denseBlock.y - 1) / denseBlock.y);
    
    matrixMultiply<<<denseGrid1, denseBlock>>>(
        nn->d_da2,             // Output gradients [BATCH_SIZE x NUM_CLASSES]
        d_W2_T,                // Transposed weights [NUM_CLASSES x HIDDEN_SIZE]
        nn->d_da1,             // Hidden layer gradients [BATCH_SIZE x HIDDEN_SIZE]
        BATCH_SIZE, HIDDEN_SIZE, NUM_CLASSES);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in dA1 calculation: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_targets);
        return;
    }
    
    // Apply ReLU derivative to get dZ1
    // dZ1 = dA1 * ReLU'(Z1)
    dim3 actBlock(256);
    dim3 actGrid((BATCH_SIZE * HIDDEN_SIZE + actBlock.x - 1) / actBlock.x);
    
    // Calculate derivative of ReLU activation
    reluDerivative<<<actGrid, actBlock>>>(
        nn->d_a1,              // Hidden layer activation [BATCH_SIZE x HIDDEN_SIZE]
        nn->d_dz1,             // Output derivative [BATCH_SIZE x HIDDEN_SIZE]
        BATCH_SIZE * HIDDEN_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in ReLU derivative kernel: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_targets);
        return;
    }
    
    // Element-wise multiply dA1 and ReLU derivative to get dZ1
    elementWiseMultiply<<<actGrid, actBlock>>>(
        nn->d_da1,             // Hidden layer gradients [BATCH_SIZE x HIDDEN_SIZE]
        nn->d_dz1,             // ReLU derivatives [BATCH_SIZE x HIDDEN_SIZE]
        nn->d_dz1,             // Output element-wise product [BATCH_SIZE x HIDDEN_SIZE]
        BATCH_SIZE * HIDDEN_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in element-wise multiplication kernel: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_targets);
        return;
    }
    
    // ===============================================================
    // Step 4: Calculate gradients for weights and biases of first layer
    // ===============================================================
    
    // Transpose text embedding for matrix multiplication
    float* d_text_embedding_T;
    CHECK_CUDA_ERROR(cudaMalloc(&d_text_embedding_T, BATCH_SIZE * EMBEDDING_DIM * sizeof(float)));
    
    dim3 transposeGrid3(
        (EMBEDDING_DIM + transposeBlock.x - 1) / transposeBlock.x,
        (BATCH_SIZE + transposeBlock.y - 1) / transposeBlock.y);
    
    matrixTranspose<<<transposeGrid3, transposeBlock>>>(
        nn->d_text_embedding,  // Text embedding [BATCH_SIZE x EMBEDDING_DIM]
        d_text_embedding_T,    // Transposed [EMBEDDING_DIM x BATCH_SIZE]
        BATCH_SIZE, EMBEDDING_DIM);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in text embedding transpose kernel: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_text_embedding_T);
        cudaFree(d_targets);
        return;
    }
    
    // Calculate dW1 = text_embedding^T * dZ1
    dim3 denseGridW1(
        (HIDDEN_SIZE + denseBlock.x - 1) / denseBlock.x,
        (EMBEDDING_DIM + denseBlock.y - 1) / denseBlock.y);
    
    // Zero out weight gradient buffer
    CHECK_CUDA_ERROR(cudaMemset(nn->d_dW1, 0, EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float)));
    
    matrixMultiply<<<denseGridW1, denseBlock>>>(
        d_text_embedding_T,    // Transposed text embedding [EMBEDDING_DIM x BATCH_SIZE]
        nn->d_dz1,             // Hidden layer gradients [BATCH_SIZE x HIDDEN_SIZE]
        nn->d_dW1,             // Weight gradients [EMBEDDING_DIM x HIDDEN_SIZE]
        EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in dW1 calculation: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_text_embedding_T);
        cudaFree(d_targets);
        return;
    }
    
    // Calculate bias gradients (db1) - sum over batch dimension
    // Similar to db2, we'll calculate on CPU for simplicity
    float* h_dz1 = new float[BATCH_SIZE * HIDDEN_SIZE];
    CHECK_CUDA_ERROR(cudaMemcpy(h_dz1, nn->d_dz1, 
                             BATCH_SIZE * HIDDEN_SIZE * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    
    float* h_db1 = new float[HIDDEN_SIZE]();
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h_db1[j] += h_dz1[b * HIDDEN_SIZE + j];
        }
    }
    
    // Copy back to device
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_db1, h_db1, 
                             HIDDEN_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    delete[] h_dz1;
    delete[] h_db1;
    
    // ===============================================================
    // Step 5: Calculate gradients for embedding layer
    // ===============================================================
    
    // Transpose W1 for matrix multiplication
    float* d_W1_T;
    CHECK_CUDA_ERROR(cudaMalloc(&d_W1_T, EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float)));
    
    dim3 transposeGrid4(
        (EMBEDDING_DIM + transposeBlock.x - 1) / transposeBlock.x,
        (HIDDEN_SIZE + transposeBlock.y - 1) / transposeBlock.y);
    
    matrixTranspose<<<transposeGrid4, transposeBlock>>>(
        nn->d_W1,              // Weights [EMBEDDING_DIM x HIDDEN_SIZE]
        d_W1_T,                // Transposed [HIDDEN_SIZE x EMBEDDING_DIM]
        EMBEDDING_DIM, HIDDEN_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in W1 transpose kernel: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_text_embedding_T);
        cudaFree(d_W1_T);
        cudaFree(d_targets);
        return;
    }
    
    // Calculate gradient for text embedding: dtext_embedding = dZ1 * W1^T
    dim3 denseGridEmbed(
        (EMBEDDING_DIM + denseBlock.x - 1) / denseBlock.x,
        (BATCH_SIZE + denseBlock.y - 1) / denseBlock.y);
    
    matrixMultiply<<<denseGridEmbed, denseBlock>>>(
        nn->d_dz1,             // Hidden layer gradients [BATCH_SIZE x HIDDEN_SIZE]
        d_W1_T,                // Transposed weights [HIDDEN_SIZE x EMBEDDING_DIM]
        nn->d_dtext_embedding, // Embedding gradients [BATCH_SIZE x EMBEDDING_DIM]
        BATCH_SIZE, EMBEDDING_DIM, HIDDEN_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in text embedding gradient calculation: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_text_embedding_T);
        cudaFree(d_W1_T);
        cudaFree(d_targets);
        return;
    }
    
    // Update the embedding vectors for words that appeared in this batch
    dim3 embedBlock(EMBEDDING_DIM);                    // Each thread handles one dimension
    dim3 embedGrid(BATCH_SIZE, MAX_SEQUENCE_LENGTH);   // Grid covers all words in batch
    
    // Zero out embedding gradients buffer
    CHECK_CUDA_ERROR(cudaMemset(nn->d_dembeddings, 0, VOCABULARY_SIZE * EMBEDDING_DIM * sizeof(float)));
    
    // Update embeddings
    updateEmbeddings<<<embedGrid, embedBlock>>>(
        nn->d_embeddings,      // Embedding matrix [VOCABULARY_SIZE x EMBEDDING_DIM]
        nn->d_dtext_embedding, // Embedding gradients [BATCH_SIZE x EMBEDDING_DIM]
        nn->d_text_indices,    // Word indices [BATCH_SIZE x MAX_SEQUENCE_LENGTH]
        LEARNING_RATE,         // Learning rate
        BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VOCABULARY_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in embedding update kernel: " 
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a1_T);
        cudaFree(d_W2_T);
        cudaFree(d_text_embedding_T);
        cudaFree(d_W1_T);
        cudaFree(d_targets);
        return;
    }
    
    // ===============================================================
    // Step 6: Update weights and biases using gradient descent
    // ===============================================================
    
    // Update W1
    dim3 updateBlockW1(256);
    dim3 updateGridW1((EMBEDDING_DIM * HIDDEN_SIZE + updateBlockW1.x - 1) / updateBlockW1.x);
    
    updateWeights<<<updateGridW1, updateBlockW1>>>(
        nn->d_W1,              // Weights [EMBEDDING_DIM x HIDDEN_SIZE]
        nn->d_dW1,             // Weight gradients [EMBEDDING_DIM x HIDDEN_SIZE]
        LEARNING_RATE,         // Learning rate
        EMBEDDING_DIM * HIDDEN_SIZE);
    
    // Update b1
    dim3 updateBlockB1(HIDDEN_SIZE);
    dim3 updateGridB1(1);
    
    updateWeights<<<updateGridB1, updateBlockB1>>>(
        nn->d_b1,              // Biases [HIDDEN_SIZE]
        nn->d_db1,             // Bias gradients [HIDDEN_SIZE]
        LEARNING_RATE,         // Learning rate
        HIDDEN_SIZE);
    
    // Update W2
    dim3 updateBlockW2(256);
    dim3 updateGridW2((HIDDEN_SIZE * NUM_CLASSES + updateBlockW2.x - 1) / updateBlockW2.x);
    
    updateWeights<<<updateGridW2, updateBlockW2>>>(
        nn->d_W2,              // Weights [HIDDEN_SIZE x NUM_CLASSES]
        nn->d_dW2,             // Weight gradients [HIDDEN_SIZE x NUM_CLASSES]
        LEARNING_RATE,         // Learning rate
        HIDDEN_SIZE * NUM_CLASSES);
    
    // Update b2
    dim3 updateBlockB2(NUM_CLASSES);
    dim3 updateGridB2(1);
    
    updateWeights<<<updateGridB2, updateBlockB2>>>(
        nn->d_b2,              // Biases [NUM_CLASSES]
        nn->d_db2,             // Bias gradients [NUM_CLASSES]
        LEARNING_RATE,         // Learning rate
        NUM_CLASSES);
    
    // Free temporary memory
    cudaFree(d_a1_T);
    cudaFree(d_W2_T);
    cudaFree(d_text_embedding_T);
    cudaFree(d_W1_T);
    cudaFree(d_targets);
    
    // Ensure all operations are complete
    cudaDeviceSynchronize();
}

// Training function for the text neural network
void trainTextNeuralNetwork(TextNeuralNetwork* nn, const std::vector<std::string>& texts, 
                         const std::vector<int>& labels, int epochs) {
    std::cout << "Starting text neural network training..." << std::endl;
    std::cout << "Training set size: " << texts.size() << " samples" << std::endl;
    std::cout << "Number of epochs: " << epochs << std::endl;
    
    // Sanity checks
    if (texts.size() != labels.size()) {
        std::cerr << "Error: Number of texts and labels don't match" << std::endl;
        return;
    }
    
    if (texts.empty()) {
        std::cerr << "Error: Empty training set" << std::endl;
        return;
    }
    
    // Make sure we have a vocabulary
    if (nn->vocabSize <= 2) {  // Just special tokens
        std::cout << "Building vocabulary from training data..." << std::endl;
        buildVocabulary(nn, texts);
    }
    
    // Allocate host memory for batch processing
    int* h_text_indices = new int[BATCH_SIZE * MAX_SEQUENCE_LENGTH];
    float* h_targets = new float[BATCH_SIZE * NUM_CLASSES];
    
    // Calculate number of batches
    int numSamples = texts.size();
    int numBatches = (numSamples + BATCH_SIZE - 1) / BATCH_SIZE;  // Ceiling division
    
    std::cout << "Training with " << numBatches << " batches per epoch" << std::endl;
    
    // Train for the specified number of epochs
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << std::endl;
        
        // Shuffle indices for this epoch
        std::vector<int> indices(numSamples);
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }
        std::random_shuffle(indices.begin(), indices.end());
        
        // Process each batch
        for (int batch = 0; batch < numBatches; batch++) {
            // Clear batch data
            std::memset(h_text_indices, 0, BATCH_SIZE * MAX_SEQUENCE_LENGTH * sizeof(int));
            std::memset(h_targets, 0, BATCH_SIZE * NUM_CLASSES * sizeof(float));
            
            // Determine actual batch size (last batch may be smaller)
            int batchStart = batch * BATCH_SIZE;
            int batchSize = std::min(BATCH_SIZE, numSamples - batchStart);
            
            // Prepare batch data
            for (int i = 0; i < batchSize; i++) {
                int idx = indices[batchStart + i];
                
                // Preprocess text to get word indices
                std::vector<std::string> tokens = tokenize(texts[idx]);
                for (int j = 0; j < tokens.size() && j < MAX_SEQUENCE_LENGTH; j++) {
                    const std::string& token = tokens[j];
                    // Look up token in vocabulary, use UNK if not found
                    int wordIdx = nn->wordToIndex.count(token) ? 
                        nn->wordToIndex[token] : nn->wordToIndex[UNK_TOKEN];
                    h_text_indices[i * MAX_SEQUENCE_LENGTH + j] = wordIdx;
                }
                
                // Pad with PAD_TOKEN
                for (int j = tokens.size(); j < MAX_SEQUENCE_LENGTH; j++) {
                    h_text_indices[i * MAX_SEQUENCE_LENGTH + j] = nn->wordToIndex[PAD_TOKEN];
                }
                
                // Create one-hot encoded target
                int label = labels[idx];
                for (int j = 0; j < NUM_CLASSES; j++) {
                    h_targets[i * NUM_CLASSES + j] = (j == label) ? 1.0f : 0.0f;
                }
            }
            
            // Forward pass
            forwardPassText(nn, h_text_indices);
            
            // Backward pass
            backwardPassText(nn, h_text_indices, h_targets);
            
            // Print progress
            if ((batch + 1) % 10 == 0 || (batch + 1) == numBatches) {
                std::cout << "  Batch " << (batch + 1) << "/" << numBatches << " processed" << std::endl;
            }
        }
    }
    
    // Free host memory
    delete[] h_text_indices;
    delete[] h_targets;
    
    std::cout << "\nTraining completed!" << std::endl;
}
