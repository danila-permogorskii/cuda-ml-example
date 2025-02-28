#include "../include/text_neural_network_utils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// Calculate the classification accuracy
float calculateAccuracy(const float* predictions, const int* labels, int numSamples) {
    int numCorrect = 0;
    
    for (int i = 0; i < numSamples; i++) {
        // Find the predicted class (index of highest probability)
        int predictedClass = 0;
        float maxProb = predictions[i * NUM_CLASSES];
        
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (predictions[i * NUM_CLASSES + j] > maxProb) {
                maxProb = predictions[i * NUM_CLASSES + j];
                predictedClass = j;
            }
        }
        
        // Check if prediction matches the true label
        if (predictedClass == labels[i]) {
            numCorrect++;
        }
    }
    
    // Return accuracy as a proportion (0 to 1)
    return static_cast<float>(numCorrect) / numSamples;
}

// Convert predicted probabilities to class labels
void predictionsToLabels(const float* predictions, int* labels, int numSamples) {
    for (int i = 0; i < numSamples; i++) {
        // Find the predicted class (index of highest probability)
        int predictedClass = 0;
        float maxProb = predictions[i * NUM_CLASSES];
        
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (predictions[i * NUM_CLASSES + j] > maxProb) {
                maxProb = predictions[i * NUM_CLASSES + j];
                predictedClass = j;
            }
        }
        
        // Store the predicted class
        labels[i] = predictedClass;
    }
}

// Print model performance metrics
void printMetrics(const float* predictions, const int* labels, int numSamples) {
    // Calculate accuracy
    float accuracy = calculateAccuracy(predictions, labels, numSamples);
    
    // For binary classification, calculate precision, recall, and F1 score
    if (NUM_CLASSES == 2) {
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < numSamples; i++) {
            // Find the predicted class
            int predictedClass = (predictions[i * NUM_CLASSES + 1] > 0.5f) ? 1 : 0;
            int trueClass = labels[i];
            
            if (predictedClass == 1 && trueClass == 1) {
                truePositives++;
            } else if (predictedClass == 1 && trueClass == 0) {
                falsePositives++;
            } else if (predictedClass == 0 && trueClass == 1) {
                falseNegatives++;
            }
        }
        
        // Calculate metrics
        float precision = (truePositives + falsePositives > 0) ? 
            static_cast<float>(truePositives) / (truePositives + falsePositives) : 0.0f;
        
        float recall = (truePositives + falseNegatives > 0) ? 
            static_cast<float>(truePositives) / (truePositives + falseNegatives) : 0.0f;
        
        float f1Score = (precision + recall > 0) ? 
            2.0f * precision * recall / (precision + recall) : 0.0f;
        
        // Print metrics
        std::cout << "Binary Classification Metrics:" << std::endl;
        std::cout << "  Accuracy:  " << std::fixed << std::setprecision(4) << accuracy << std::endl;
        std::cout << "  Precision: " << std::fixed << std::setprecision(4) << precision << std::endl;
        std::cout << "  Recall:    " << std::fixed << std::setprecision(4) << recall << std::endl;
        std::cout << "  F1 Score:  " << std::fixed << std::setprecision(4) << f1Score << std::endl;
    } else {
        // For multi-class classification, just print accuracy
        std::cout << "Multi-class Classification Metrics:" << std::endl;
        std::cout << "  Accuracy: " << std::fixed << std::setprecision(4) << accuracy << std::endl;
    }
}

// Print a confusion matrix for binary classification
void printConfusionMatrix(const float* predictions, const int* labels, int numSamples) {
    if (NUM_CLASSES != 2) {
        std::cout << "Confusion matrix is only supported for binary classification." << std::endl;
        return;
    }
    
    int truePositives = 0;
    int falsePositives = 0;
    int trueNegatives = 0;
    int falseNegatives = 0;
    
    for (int i = 0; i < numSamples; i++) {
        // Find the predicted class
        int predictedClass = (predictions[i * NUM_CLASSES + 1] > 0.5f) ? 1 : 0;
        int trueClass = labels[i];
        
        if (predictedClass == 1 && trueClass == 1) {
            truePositives++;
        } else if (predictedClass == 1 && trueClass == 0) {
            falsePositives++;
        } else if (predictedClass == 0 && trueClass == 0) {
            trueNegatives++;
        } else if (predictedClass == 0 && trueClass == 1) {
            falseNegatives++;
        }
    }
    
    // Print confusion matrix
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << std::setw(14) << "Predicted 0" << std::setw(14) << "Predicted 1" << std::endl;
    std::cout << "Actual 0  " << std::setw(14) << trueNegatives << std::setw(14) << falsePositives << std::endl;
    std::cout << "Actual 1  " << std::setw(14) << falseNegatives << std::setw(14) << truePositives << std::endl;
}

// Save trained model weights to files
void saveModelWeights(TextNeuralNetwork* nn, const std::string& prefix) {
    std::cout << "Saving model weights to " << prefix << "*.bin files..." << std::endl;
    
    // Allocate host memory for weights and biases
    float* h_embeddings = new float[VOCABULARY_SIZE * EMBEDDING_DIM];
    float* h_W1 = new float[EMBEDDING_DIM * HIDDEN_SIZE];
    float* h_b1 = new float[HIDDEN_SIZE];
    float* h_W2 = new float[HIDDEN_SIZE * NUM_CLASSES];
    float* h_b2 = new float[NUM_CLASSES];
    
    // Copy weights from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_embeddings, nn->d_embeddings, 
                             VOCABULARY_SIZE * EMBEDDING_DIM * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_W1, nn->d_W1, 
                             EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_b1, nn->d_b1, 
                             HIDDEN_SIZE * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_W2, nn->d_W2, 
                             HIDDEN_SIZE * NUM_CLASSES * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_b2, nn->d_b2, 
                             NUM_CLASSES * sizeof(float), 
                             cudaMemcpyDeviceToHost));
    
    // Save embeddings
    std::ofstream embFile(prefix + "embeddings.bin", std::ios::binary);
    if (embFile.is_open()) {
        embFile.write(reinterpret_cast<char*>(h_embeddings), 
                     VOCABULARY_SIZE * EMBEDDING_DIM * sizeof(float));
        embFile.close();
    } else {
        std::cerr << "Error: Could not open file for writing embeddings" << std::endl;
    }
    
    // Save W1
    std::ofstream w1File(prefix + "W1.bin", std::ios::binary);
    if (w1File.is_open()) {
        w1File.write(reinterpret_cast<char*>(h_W1), 
                    EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float));
        w1File.close();
    } else {
        std::cerr << "Error: Could not open file for writing W1" << std::endl;
    }
    
    // Save b1
    std::ofstream b1File(prefix + "b1.bin", std::ios::binary);
    if (b1File.is_open()) {
        b1File.write(reinterpret_cast<char*>(h_b1), 
                    HIDDEN_SIZE * sizeof(float));
        b1File.close();
    } else {
        std::cerr << "Error: Could not open file for writing b1" << std::endl;
    }
    
    // Save W2
    std::ofstream w2File(prefix + "W2.bin", std::ios::binary);
    if (w2File.is_open()) {
        w2File.write(reinterpret_cast<char*>(h_W2), 
                    HIDDEN_SIZE * NUM_CLASSES * sizeof(float));
        w2File.close();
    } else {
        std::cerr << "Error: Could not open file for writing W2" << std::endl;
    }
    
    // Save b2
    std::ofstream b2File(prefix + "b2.bin", std::ios::binary);
    if (b2File.is_open()) {
        b2File.write(reinterpret_cast<char*>(h_b2), 
                    NUM_CLASSES * sizeof(float));
        b2File.close();
    } else {
        std::cerr << "Error: Could not open file for writing b2" << std::endl;
    }
    
    // Free host memory
    delete[] h_embeddings;
    delete[] h_W1;
    delete[] h_b1;
    delete[] h_W2;
    delete[] h_b2;
    
    std::cout << "Model weights saved successfully!" << std::endl;
}

// Load model weights from files
void loadModelWeights(TextNeuralNetwork* nn, const std::string& prefix) {
    std::cout << "Loading model weights from " << prefix << "*.bin files..." << std::endl;
    
    // Allocate host memory for weights and biases
    float* h_embeddings = new float[VOCABULARY_SIZE * EMBEDDING_DIM];
    float* h_W1 = new float[EMBEDDING_DIM * HIDDEN_SIZE];
    float* h_b1 = new float[HIDDEN_SIZE];
    float* h_W2 = new float[HIDDEN_SIZE * NUM_CLASSES];
    float* h_b2 = new float[NUM_CLASSES];
    
    // Load embeddings
    std::ifstream embFile(prefix + "embeddings.bin", std::ios::binary);
    if (embFile.is_open()) {
        embFile.read(reinterpret_cast<char*>(h_embeddings), 
                    VOCABULARY_SIZE * EMBEDDING_DIM * sizeof(float));
        embFile.close();
    } else {
        std::cerr << "Error: Could not open file for reading embeddings" << std::endl;
        delete[] h_embeddings;
        delete[] h_W1;
        delete[] h_b1;
        delete[] h_W2;
        delete[] h_b2;
        return;
    }
    
    // Load W1
    std::ifstream w1File(prefix + "W1.bin", std::ios::binary);
    if (w1File.is_open()) {
        w1File.read(reinterpret_cast<char*>(h_W1), 
                   EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float));
        w1File.close();
    } else {
        std::cerr << "Error: Could not open file for reading W1" << std::endl;
        delete[] h_embeddings;
        delete[] h_W1;
        delete[] h_b1;
        delete[] h_W2;
        delete[] h_b2;
        return;
    }
    
    // Load b1
    std::ifstream b1File(prefix + "b1.bin", std::ios::binary);
    if (b1File.is_open()) {
        b1File.read(reinterpret_cast<char*>(h_b1), 
                   HIDDEN_SIZE * sizeof(float));
        b1File.close();
    } else {
        std::cerr << "Error: Could not open file for reading b1" << std::endl;
        delete[] h_embeddings;
        delete[] h_W1;
        delete[] h_b1;
        delete[] h_W2;
        delete[] h_b2;
        return;
    }
    
    // Load W2
    std::ifstream w2File(prefix + "W2.bin", std::ios::binary);
    if (w2File.is_open()) {
        w2File.read(reinterpret_cast<char*>(h_W2), 
                   HIDDEN_SIZE * NUM_CLASSES * sizeof(float));
        w2File.close();
    } else {
        std::cerr << "Error: Could not open file for reading W2" << std::endl;
        delete[] h_embeddings;
        delete[] h_W1;
        delete[] h_b1;
        delete[] h_W2;
        delete[] h_b2;
        return;
    }
    
    // Load b2
    std::ifstream b2File(prefix + "b2.bin", std::ios::binary);
    if (b2File.is_open()) {
        b2File.read(reinterpret_cast<char*>(h_b2), 
                   NUM_CLASSES * sizeof(float));
        b2File.close();
    } else {
        std::cerr << "Error: Could not open file for reading b2" << std::endl;
        delete[] h_embeddings;
        delete[] h_W1;
        delete[] h_b1;
        delete[] h_W2;
        delete[] h_b2;
        return;
    }
    
    // Copy weights from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_embeddings, h_embeddings, 
                             VOCABULARY_SIZE * EMBEDDING_DIM * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_W1, h_W1, 
                             EMBEDDING_DIM * HIDDEN_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_b1, h_b1, 
                             HIDDEN_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_W2, h_W2, 
                             HIDDEN_SIZE * NUM_CLASSES * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    CHECK_CUDA_ERROR(cudaMemcpy(nn->d_b2, h_b2, 
                             NUM_CLASSES * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    // Free host memory
    delete[] h_embeddings;
    delete[] h_W1;
    delete[] h_b1;
    delete[] h_W2;
    delete[] h_b2;
    
    std::cout << "Model weights loaded successfully!" << std::endl;
}

// Save vocabulary to a file
void saveVocabulary(TextNeuralNetwork* nn, const std::string& filename) {
    std::cout << "Saving vocabulary to " << filename << "..." << std::endl;
    
    std::ofstream vocabFile(filename);
    if (!vocabFile.is_open()) {
        std::cerr << "Error: Could not open file for writing vocabulary" << std::endl;
        return;
    }
    
    // First line: vocabulary size
    vocabFile << nn->vocabSize << std::endl;
    
    // Remaining lines: index word
    for (int i = 0; i < nn->vocabSize; i++) {
        vocabFile << i << " " << nn->indexToWord[i] << std::endl;
    }
    
    vocabFile.close();
    std::cout << "Vocabulary saved successfully!" << std::endl;
}

// Load vocabulary from a file
void loadVocabulary(TextNeuralNetwork* nn, const std::string& filename) {
    std::cout << "Loading vocabulary from " << filename << "..." << std::endl;
    
    std::ifstream vocabFile(filename);
    if (!vocabFile.is_open()) {
        std::cerr << "Error: Could not open file for reading vocabulary" << std::endl;
        return;
    }
    
    // Clear existing vocabulary
    nn->wordToIndex.clear();
    nn->indexToWord.clear();
    
    // First line: vocabulary size
    int vocabSize;
    vocabFile >> vocabSize;
    
    // Read each word
    for (int i = 0; i < vocabSize; i++) {
        int idx;
        std::string word;
        vocabFile >> idx >> word;
        
        nn->wordToIndex[word] = idx;
        nn->indexToWord.push_back(word);
    }
    
    nn->vocabSize = vocabSize;
    
    vocabFile.close();
    std::cout << "Loaded vocabulary with " << nn->vocabSize << " words" << std::endl;
}
