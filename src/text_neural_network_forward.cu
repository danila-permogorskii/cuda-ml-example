#include "../include/text_neural_network.h"
#include "../include/text_neural_network_kernels.cuh"
#include <iostream>

// Forward pass for text neural network
void forwardPassText(TextNeuralNetwork* nn, int* textIndices) {
    // Step 0: Copy text indices to device if they're not already there
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_text_indices, textIndices, 
                             BATCH_SIZE * MAX_SEQUENCE_LENGTH * sizeof(int), 
                             cudaMemcpyHostToDevice));
    
    // Step 1: Lookup embeddings for each word
    // Each thread handles one embedding dimension for one word
    dim3 embLookupBlock(EMBEDDING_DIM); // Thread block dimensions
    dim3 embLookupGrid(BATCH_SIZE, MAX_SEQUENCE_LENGTH); // Grid dimensions
    
    lookupEmbeddings<<<embLookupGrid, embLookupBlock>>>(
        nn->d_text_indices,  // Input word indices
        nn->d_embeddings,    // Embedding matrix
        nn->d_word_embeddings, // Output word embeddings
        BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VOCABULARY_SIZE);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in lookupEmbeddings kernel launch: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Step 2: Average word embeddings to get text embedding
    // Each thread handles one embedding dimension for one text sample
    dim3 embAvgBlock(EMBEDDING_DIM); // Thread block dimensions
    dim3 embAvgGrid(BATCH_SIZE);     // Grid dimensions
    
    averageEmbeddings<<<embAvgGrid, embAvgBlock>>>(
        nn->d_word_embeddings,  // Input word embeddings
        nn->d_text_embedding,   // Output text embedding
        BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in averageEmbeddings kernel launch: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Step 3: First dense layer (text embedding -> hidden)
    // Each thread computes one element of the output matrix
    dim3 denseBlock(16, 16);    // Thread block dimensions (16x16 threads)
    dim3 denseGrid1(
        (HIDDEN_SIZE + denseBlock.x - 1) / denseBlock.x,  // Ceiling division for grid x dimension
        (BATCH_SIZE + denseBlock.y - 1) / denseBlock.y);  // Ceiling division for grid y dimension
    
    // Z1 = TextEmbed * W1
    matrixMultiply<<<denseGrid1, denseBlock>>>(
        nn->d_text_embedding,  // Input text embedding [BATCH_SIZE x EMBEDDING_DIM]
        nn->d_W1,              // Weights [EMBEDDING_DIM x HIDDEN_SIZE]
        nn->d_z1,              // Output pre-activation [BATCH_SIZE x HIDDEN_SIZE]
        BATCH_SIZE, HIDDEN_SIZE, EMBEDDING_DIM);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in first matrixMultiply kernel launch: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Z1 = Z1 + b1
    addBiasToMatrix<<<denseGrid1, denseBlock>>>(
        nn->d_z1,              // Input/output pre-activation [BATCH_SIZE x HIDDEN_SIZE]
        nn->d_b1,              // Bias vector [HIDDEN_SIZE]
        BATCH_SIZE, HIDDEN_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in first addBiasToMatrix kernel launch: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // A1 = ReLU(Z1)
    // Each thread applies ReLU to one element
    dim3 actBlock(256);        // Thread block dimension
    dim3 actGrid((BATCH_SIZE * HIDDEN_SIZE + actBlock.x - 1) / actBlock.x); // Grid dimension
    
    reluActivation<<<actGrid, actBlock>>>(
        nn->d_z1,              // Input/output values [BATCH_SIZE * HIDDEN_SIZE]
        BATCH_SIZE * HIDDEN_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in reluActivation kernel launch: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Copy Z1 to A1 (since ReLU is applied in-place to Z1)
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_a1, nn->d_z1, 
                             BATCH_SIZE * HIDDEN_SIZE * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
    
    // Step 4: Second dense layer (hidden -> output)
    dim3 denseGrid2(
        (NUM_CLASSES + denseBlock.x - 1) / denseBlock.x,  // Grid x dimension
        (BATCH_SIZE + denseBlock.y - 1) / denseBlock.y);  // Grid y dimension
    
    // Z2 = A1 * W2
    matrixMultiply<<<denseGrid2, denseBlock>>>(
        nn->d_a1,              // Input hidden activation [BATCH_SIZE x HIDDEN_SIZE]
        nn->d_W2,              // Weights [HIDDEN_SIZE x NUM_CLASSES]
        nn->d_z2,              // Output pre-activation [BATCH_SIZE x NUM_CLASSES]
        BATCH_SIZE, NUM_CLASSES, HIDDEN_SIZE);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in second matrixMultiply kernel launch: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Z2 = Z2 + b2
    addBiasToMatrix<<<denseGrid2, denseBlock>>>(
        nn->d_z2,              // Input/output pre-activation [BATCH_SIZE x NUM_CLASSES]
        nn->d_b2,              // Bias vector [NUM_CLASSES]
        BATCH_SIZE, NUM_CLASSES);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in second addBiasToMatrix kernel launch: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // A2 = Softmax(Z2)
    // Each block handles one sample
    dim3 softmaxGrid(BATCH_SIZE);  // Grid dimension
    dim3 softmaxBlock(1);          // Thread block dimension
    
    softmaxActivation<<<softmaxGrid, softmaxBlock>>>(
        nn->d_z2,              // Input/output pre-activation [BATCH_SIZE x NUM_CLASSES]
        BATCH_SIZE, NUM_CLASSES);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in softmaxActivation kernel launch: " 
                 << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Copy Z2 to A2 (since softmax is applied in-place to Z2)
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_a2, nn->d_z2, 
                             BATCH_SIZE * NUM_CLASSES * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
    
    // Ensure all operations have completed
    cudaDeviceSynchronize();
}

// Single sample prediction function
void predictSentiment(TextNeuralNetwork* nn, const std::string& text, float* output) {
    // Tokenize the input text
    std::vector<std::string> tokens = tokenize(text);
    
    std::cout << "Processing text: " << text << std::endl;
    std::cout << "Tokenized " << tokens.size() << " words" << std::endl;
    
    // Prepare text indices
    int* h_indices = new int[MAX_SEQUENCE_LENGTH]();
    
    // Convert tokens to indices
    for (int i = 0; i < tokens.size() && i < MAX_SEQUENCE_LENGTH; i++) {
        const std::string& token = tokens[i];
        // Look up token in vocabulary, use UNK if not found
        h_indices[i] = nn->wordToIndex.count(token) ? 
            nn->wordToIndex[token] : nn->wordToIndex[UNK_TOKEN];
    }
    
    // Pad with PAD_TOKEN
    for (int i = tokens.size(); i < MAX_SEQUENCE_LENGTH; i++) {
        h_indices[i] = nn->wordToIndex[PAD_TOKEN];
    }
    
    // Allocate batch of indices (we'll only use the first sample)
    int* h_batch_indices = new int[BATCH_SIZE * MAX_SEQUENCE_LENGTH]();
    
    // Copy indices to the first sample in the batch
    for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++) {
        h_batch_indices[i] = h_indices[i];
    }
    
    // Forward pass
    forwardPassText(nn, h_batch_indices);
    
    // Copy output back to host
    float* h_output = new float[BATCH_SIZE * NUM_CLASSES];
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, nn->d_a2, 
                             BATCH_SIZE * NUM_CLASSES * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    
    // Copy results to output (first sample only)
    for (int i = 0; i < NUM_CLASSES; i++) {
        output[i] = h_output[i];
    }
    
    // Free temporary memory
    delete[] h_indices;
    delete[] h_batch_indices;
    delete[] h_output;
    
    std::cout << "Prediction complete" << std::endl;
    std::cout << "Classes probabilities: ";
    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << "Class " << i << ": " << output[i] << " ";
    }
    std::cout << std::endl;
}
