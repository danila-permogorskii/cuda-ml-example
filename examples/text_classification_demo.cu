#include "../include/text_neural_network.h"
#include "../include/text_neural_network_utils.h"
#include "../include/text_preprocessing.h"
#include "../include/cuda_ml_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>  // For std::memset

// Forward declaration of forwardPassText since it's implemented in another file
void forwardPassText(TextNeuralNetwork* nn, int* textIndices);

// Function to generate a simple sentiment dataset
void generateDataset(std::vector<std::string>& trainTexts, std::vector<int>& trainLabels,
                    std::vector<std::string>& testTexts, std::vector<int>& testLabels,
                    int numTrainSamples, int numTestSamples) {
    
    std::vector<std::string> allTexts;
    std::vector<int> allLabels;
    
    // Generate synthetic data
    generateSimpleSentimentDataset(allTexts, allLabels, numTrainSamples + numTestSamples);
    
    // Shuffle the data
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::vector<size_t> indices(allTexts.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
    
    // Split into train and test sets
    trainTexts.clear();
    trainLabels.clear();
    testTexts.clear();
    testLabels.clear();
    
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (i < numTrainSamples) {
            trainTexts.push_back(allTexts[idx]);
            trainLabels.push_back(allLabels[idx]);
        } else {
            testTexts.push_back(allTexts[idx]);
            testLabels.push_back(allLabels[idx]);
        }
    }
}

// Function to evaluate model on test set
void evaluateModel(TextNeuralNetwork* nn, const std::vector<std::string>& testTexts,
                 const std::vector<int>& testLabels) {
    
    std::cout << "\nEvaluating model on test set..." << std::endl;
    
    // Calculate how many batches we need
    int numTestSamples = testTexts.size();
    int numBatches = (numTestSamples + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // Allocate memory for indices and predictions
    int* h_text_indices = new int[BATCH_SIZE * MAX_SEQUENCE_LENGTH];
    float* h_predictions = new float[BATCH_SIZE * NUM_CLASSES];
    int* h_batch_predictions = new int[BATCH_SIZE];
    int* h_batch_labels = new int[BATCH_SIZE];
    
    std::vector<float> allPredictions;
    std::vector<int> allLabels;
    
    // Process each batch
    for (int batch = 0; batch < numBatches; batch++) {
        // Determine actual batch size (last batch may be smaller)
        int batchStart = batch * BATCH_SIZE;
        int batchSize = std::min(BATCH_SIZE, numTestSamples - batchStart);
        
        // Prepare batch data
        for (int i = 0; i < batchSize; i++) {
            // Preprocess text
            std::vector<std::string> tokens = tokenize(testTexts[batchStart + i]);
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
            
            // Store the label
            h_batch_labels[i] = testLabels[batchStart + i];
        }
        
        // Forward pass
        forwardPassText(nn, h_text_indices);
        
        // Copy predictions from device to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, nn->d_a2, 
                                 BATCH_SIZE * NUM_CLASSES * sizeof(float), 
                                 cudaMemcpyDeviceToHost));
        
        // Convert to predicted labels
        predictionsToLabels(h_predictions, h_batch_predictions, batchSize);
        
        // Collect predictions and labels
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < NUM_CLASSES; j++) {
                allPredictions.push_back(h_predictions[i * NUM_CLASSES + j]);
            }
            allLabels.push_back(h_batch_labels[i]);
        }
    }
    
    // Calculate metrics
    printMetrics(allPredictions.data(), allLabels.data(), allLabels.size());
    
    // Print confusion matrix
    printConfusionMatrix(allPredictions.data(), allLabels.data(), allLabels.size());
    
    // Clean up
    delete[] h_text_indices;
    delete[] h_predictions;
    delete[] h_batch_predictions;
    delete[] h_batch_labels;
}

// Main function to demonstrate text classification
int main() {
    std::cout << "CUDA Text Classification Example" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Print device information
    printDeviceInfo();
    
    // Create and initialize the neural network
    TextNeuralNetwork nn;
    initializeTextNeuralNetwork(&nn);
    
    // Generate synthetic dataset
    std::vector<std::string> trainTexts;
    std::vector<int> trainLabels;
    std::vector<std::string> testTexts;
    std::vector<int> testLabels;
    
    int numTrainSamples = 800;
    int numTestSamples = 200;
    
    std::cout << "\nGenerating synthetic dataset..." << std::endl;
    generateDataset(trainTexts, trainLabels, testTexts, testLabels, 
                   numTrainSamples, numTestSamples);
    
    std::cout << "Training set size: " << trainTexts.size() << " samples" << std::endl;
    std::cout << "Test set size: " << testTexts.size() << " samples" << std::endl;
    
    // Train the model
    std::cout << "\nTraining text neural network..." << std::endl;
    
    // Let's use a timer to measure training time
    CudaTimer timer;
    timer.initialize();
    timer.startTimer();
    
    // Train for 5 epochs
    trainTextNeuralNetwork(&nn, trainTexts, trainLabels, 5);
    
    float trainingTime = timer.stopTimer();
    std::cout << "Training completed in " << trainingTime / 1000.0f << " seconds" << std::endl;
    
    // Evaluate on test set
    evaluateModel(&nn, testTexts, testLabels);
    
    // Test with custom examples
    std::cout << "\nTesting with custom examples:" << std::endl;
    
    std::vector<std::string> customExamples = {
        "this is really good and fantastic",
        "that was terrible and disappointing",
        "i am satisfied with the service",
        "the product was awful and poor quality"
    };
    
    for (const auto& example : customExamples) {
        float predictions[NUM_CLASSES];
        predictSentiment(&nn, example, predictions);
        
        std::cout << "Text: \"" << example << "\"" << std::endl;
        std::cout << "  Prediction: ";
        
        if (predictions[1] > predictions[0]) {
            std::cout << "Positive (" << predictions[1] * 100.0f << "% confidence)" << std::endl;
        } else {
            std::cout << "Negative (" << predictions[0] * 100.0f << "% confidence)" << std::endl;
        }
    }
    
    // Create directory for models if it doesn't exist
    #ifdef _WIN32
    system("mkdir models 2>nul");
    #else
    system("mkdir -p models");
    #endif
    
    // Save the model
    std::cout << "\nSaving model..." << std::endl;
    saveModelWeights(&nn, "models/sentiment_");
    saveVocabulary(&nn, "models/sentiment_vocab.txt");
    
    // Free neural network resources
    freeTextNeuralNetwork(&nn);
    
    std::cout << "\nText classification demo completed!" << std::endl;
    
    return 0;
}
