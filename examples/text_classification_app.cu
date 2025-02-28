#include "../include/text_neural_network.h"
#include "../include/text_neural_network_utils.h"
#include "../include/text_preprocessing.h"
#include "../include/cuda_ml_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <limits>
#include <filesystem>

// Function to print a fancy header
void printHeader(const std::string& title) {
    int width = 60;
    std::string border(width, '=');
    
    std::cout << "\n" << border << std::endl;
    int padding = (width - title.length()) / 2;
    std::cout << std::string(padding, ' ') << title << std::endl;
    std::cout << border << "\n" << std::endl;
}

// Function to get user input with validation
template<typename T>
T getValidInput(const std::string& prompt, T min_val, T max_val) {
    T value;
    bool validInput = false;
    
    do {
        std::cout << prompt;
        if (std::cin >> value) {
            if (value >= min_val && value <= max_val) {
                validInput = true;
            } else {
                std::cout << "Error: Please enter a value between " << min_val << " and " << max_val << std::endl;
            }
        } else {
            std::cout << "Error: Invalid input, please try again." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    } while (!validInput);
    
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return value;
}

// Function to get string input
std::string getStringInput(const std::string& prompt) {
    std::string input;
    std::cout << prompt;
    std::getline(std::cin, input);
    return input;
}

// Function to train the model on synthetic data
void trainModelOption(TextNeuralNetwork* nn) {
    printHeader("TRAIN NEW MODEL");
    
    // Get parameters from user
    int trainSize = getValidInput<int>("Enter number of training samples (100-5000): ", 100, 5000);
    int testSize = getValidInput<int>("Enter number of test samples (10-1000): ", 10, 1000);
    int epochs = getValidInput<int>("Enter number of training epochs (1-20): ", 1, 20);
    
    std::cout << "\nGenerating synthetic dataset..." << std::endl;
    
    // Generate synthetic dataset
    std::vector<std::string> trainTexts;
    std::vector<int> trainLabels;
    std::vector<std::string> testTexts;
    std::vector<int> testLabels;
    
    // Generate all data at once
    std::vector<std::string> allTexts;
    std::vector<int> allLabels;
    generateSimpleSentimentDataset(allTexts, allLabels, trainSize + testSize);
    
    // Shuffle and split
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<size_t> indices(allTexts.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
    
    // Split into train and test sets
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (i < trainSize) {
            trainTexts.push_back(allTexts[idx]);
            trainLabels.push_back(allLabels[idx]);
        } else if (i < trainSize + testSize) {
            testTexts.push_back(allTexts[idx]);
            testLabels.push_back(allLabels[idx]);
        }
    }
    
    std::cout << "Training set size: " << trainTexts.size() << " samples" << std::endl;
    std::cout << "Test set size: " << testTexts.size() << " samples" << std::endl;
    
    // Train the model
    std::cout << "\nTraining model..." << std::endl;
    
    // Use timer to measure training time
    CudaTimer timer;
    timer.initialize();
    timer.startTimer();
    
    // Train for specified epochs
    trainTextNeuralNetwork(nn, trainTexts, trainLabels, epochs);
    
    float trainingTime = timer.stopTimer();
    std::cout << "Training completed in " << trainingTime / 1000.0f << " seconds" << std::endl;
    
    // Evaluate on test set
    std::cout << "\nEvaluating model on test set..." << std::endl;
    
    // Allocate memory for predictions
    int* h_text_indices = new int[BATCH_SIZE * MAX_SEQUENCE_LENGTH];
    float* h_predictions = new float[BATCH_SIZE * NUM_CLASSES];
    int* h_batch_predictions = new int[BATCH_SIZE];
    int* h_batch_labels = new int[BATCH_SIZE];
    
    std::vector<float> allPredictions;
    std::vector<int> allLabels;
    
    // Process test data in batches
    int numBatches = (testSize + BATCH_SIZE - 1) / BATCH_SIZE;
    for (int batch = 0; batch < numBatches; batch++) {
        int batchStart = batch * BATCH_SIZE;
        int batchSize = std::min(BATCH_SIZE, testSize - batchStart);
        
        // Process each sample in batch
        for (int i = 0; i < batchSize; i++) {
            std::vector<std::string> tokens = tokenize(testTexts[batchStart + i]);
            for (int j = 0; j < tokens.size() && j < MAX_SEQUENCE_LENGTH; j++) {
                const std::string& token = tokens[j];
                int wordIdx = nn->wordToIndex.count(token) ? 
                    nn->wordToIndex[token] : nn->wordToIndex[UNK_TOKEN];
                h_text_indices[i * MAX_SEQUENCE_LENGTH + j] = wordIdx;
            }
            
            // Pad sequence
            for (int j = tokens.size(); j < MAX_SEQUENCE_LENGTH; j++) {
                h_text_indices[i * MAX_SEQUENCE_LENGTH + j] = nn->wordToIndex[PAD_TOKEN];
            }
            
            h_batch_labels[i] = testLabels[batchStart + i];
        }
        
        // Forward pass
        forwardPassText(nn, h_text_indices);
        
        // Get predictions
        CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, nn->d_a2, 
                                 BATCH_SIZE * NUM_CLASSES * sizeof(float), 
                                 cudaMemcpyDeviceToHost));
        
        // Convert to class labels
        predictionsToLabels(h_predictions, h_batch_predictions, batchSize);
        
        // Collect results
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < NUM_CLASSES; j++) {
                allPredictions.push_back(h_predictions[i * NUM_CLASSES + j]);
            }
            allLabels.push_back(h_batch_labels[i]);
        }
    }
    
    // Print performance metrics
    printMetrics(allPredictions.data(), allLabels.data(), allLabels.size());
    printConfusionMatrix(allPredictions.data(), allLabels.data(), allLabels.size());
    
    // Clean up
    delete[] h_text_indices;
    delete[] h_predictions;
    delete[] h_batch_predictions;
    delete[] h_batch_labels;
    
    // Ask if user wants to save the model
    std::string saveResponse = getStringInput("\nDo you want to save this model? (y/n): ");
    if (saveResponse == "y" || saveResponse == "Y") {
        // Create models directory if it doesn't exist
        #ifdef _WIN32
        system("mkdir models 2>nul");
        #else
        system("mkdir -p models");
        #endif
        
        std::string modelName = getStringInput("Enter model name (without extension): ");
        if (modelName.empty()) {
            modelName = "sentiment_model";
        }
        
        std::string prefix = "models/" + modelName + "_";
        saveModelWeights(nn, prefix);
        saveVocabulary(nn, prefix + "vocab.txt");
        
        std::cout << "Model saved to: " << prefix << "*.bin" << std::endl;
    }
    
    std::cout << "Press Enter to continue...";
    std::cin.get();
}

// Function to load a pre-trained model
bool loadModelOption(TextNeuralNetwork* nn) {
    printHeader("LOAD EXISTING MODEL");
    
    // Check if models directory exists
    #ifdef _WIN32
    if (system("dir models >nul 2>&1") != 0) {
    #else
    if (system("test -d models") != 0) {
    #endif
        std::cout << "No models directory found." << std::endl;
        std::cout << "Press Enter to continue...";
        std::cin.get();
        return false;
    }
    
    // List available models
    std::cout << "Available models:" << std::endl;
    
    std::vector<std::string> modelNames;
    
    // Use filesystem to list files, this requires C++17
    #ifdef _WIN32
    std::string dirCommand = "dir /b models\\*_vocab.txt >models\\model_list.txt";
    system(dirCommand.c_str());
    FILE* modelList = fopen("models\\model_list.txt", "r");
    #else
    std::string dirCommand = "ls -1 models/*_vocab.txt > models/model_list.txt";
    system(dirCommand.c_str());
    FILE* modelList = fopen("models/model_list.txt", "r");
    #endif
    
    if (modelList != nullptr) {
        char line[256];
        int index = 1;
        
        while (fgets(line, sizeof(line), modelList)) {
            // Remove newline character
            size_t len = strlen(line);
            if (len > 0 && line[len-1] == '\n') {
                line[len-1] = '\0';
            }
            
            // Extract model name from vocab file
            std::string vocabFile = line;
            // Remove "models/" or "models\"
            #ifdef _WIN32
            size_t startPos = 0; // Already stripped by dir /b command
            #else
            size_t startPos = vocabFile.find("/") + 1;
            #endif
            // Remove "_vocab.txt"
            size_t endPos = vocabFile.rfind("_vocab.txt");
            if (endPos != std::string::npos) {
                std::string modelName = vocabFile.substr(startPos, endPos - startPos);
                modelNames.push_back(modelName);
                std::cout << "  " << index++ << ". " << modelName << std::endl;
            }
        }
        fclose(modelList);
    }
    
    #ifdef _WIN32
    system("del models\\model_list.txt >nul 2>&1");
    #else
    system("rm models/model_list.txt");
    #endif
    
    if (modelNames.empty()) {
        std::cout << "No models found." << std::endl;
        std::cout << "Press Enter to continue...";
        std::cin.get();
        return false;
    }
    
    // Get user selection
    int modelIndex = getValidInput<int>("Enter model number to load: ", 1, static_cast<int>(modelNames.size()));
    std::string selectedModel = modelNames[modelIndex - 1];
    
    // Load the model
    std::string prefix = "models/" + selectedModel + "_";
    
    // First load vocabulary as it's needed for the network structure
    loadVocabulary(nn, prefix + "vocab.txt");
    
    // Then load weights
    loadModelWeights(nn, prefix);
    
    std::cout << "Model '" << selectedModel << "' loaded successfully!" << std::endl;
    std::cout << "Press Enter to continue...";
    std::cin.get();
    
    return true;
}

// Function to analyze text and explain the prediction
void analyzeTextOption(TextNeuralNetwork* nn) {
    printHeader("ANALYZE TEXT");
    
    // Check if model is initialized
    if (nn->vocabSize <= 2) {  // Just special tokens
        std::cout << "No model loaded. Please train or load a model first." << std::endl;
        std::cout << "Press Enter to continue...";
        std::cin.get();
        return;
    }
    
    while (true) {
        // Get text input
        std::string text = getStringInput("Enter text to analyze (or type 'exit' to return to menu): ");
        if (text == "exit") {
            break;
        }
        
        // Make prediction
        float predictions[NUM_CLASSES];
        predictSentiment(nn, text, predictions);
        
        // Display prediction and confidence
        std::cout << "\nAnalysis Results:" << std::endl;
        std::cout << "----------------------------" << std::endl;
        
        // For binary sentiment classification
        if (NUM_CLASSES == 2) {
            float confidence = std::max(predictions[0], predictions[1]) * 100.0f;
            
            std::cout << "Text: \"" << text << "\"" << std::endl;
            std::cout << "Sentiment: ";
            
            if (predictions[1] > predictions[0]) {
                std::cout << "POSITIVE with " << std::fixed << std::setprecision(1) 
                         << confidence << "% confidence" << std::endl;
            } else {
                std::cout << "NEGATIVE with " << std::fixed << std::setprecision(1) 
                         << confidence << "% confidence" << std::endl;
            }
            
            // Explain the prediction
            std::cout << "\nSignal words detected:" << std::endl;
            std::vector<std::string> tokens = tokenize(text);
            
            // Use a simple approach to highlight words that might have influenced the prediction
            std::vector<std::pair<std::string, std::string>> sentimentWords = {
                {"good", "positive"},
                {"great", "positive"},
                {"excellent", "positive"},
                {"wonderful", "positive"},
                {"amazing", "positive"},
                {"fantastic", "positive"},
                {"love", "positive"},
                {"happy", "positive"},
                {"bad", "negative"},
                {"terrible", "negative"},
                {"awful", "negative"},
                {"horrible", "negative"},
                {"disappointing", "negative"},
                {"poor", "negative"},
                {"hate", "negative"},
                {"dislike", "negative"}
            };
            
            bool foundWords = false;
            for (const auto& token : tokens) {
                for (const auto& sentimentPair : sentimentWords) {
                    if (token.find(sentimentPair.first) != std::string::npos) {
                        std::cout << "  - \"" << token << "\" (usually " << sentimentPair.second << ")" << std::endl;
                        foundWords = true;
                    }
                }
            }
            
            if (!foundWords) {
                std::cout << "  No definitive sentiment words detected." << std::endl;
                std::cout << "  The classification may be based on subtle patterns learned during training." << std::endl;
            }
        } else {
            // For multi-class classification, show probabilities for all classes
            std::cout << "Text: \"" << text << "\"" << std::endl;
            std::cout << "Classification results:" << std::endl;
            
            int maxClass = 0;
            for (int i = 1; i < NUM_CLASSES; i++) {
                if (predictions[i] > predictions[maxClass]) {
                    maxClass = i;
                }
            }
            
            for (int i = 0; i < NUM_CLASSES; i++) {
                std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(1) 
                         << predictions[i] * 100.0f << "%" 
                         << (i == maxClass ? " (SELECTED)" : "") << std::endl;
            }
        }
        
        std::cout << std::endl;
    }
}

// Function to show information about the model
void showModelInfoOption(TextNeuralNetwork* nn) {
    printHeader("MODEL INFORMATION");
    
    // Check if model is initialized
    if (nn->vocabSize <= 2) {  // Just special tokens
        std::cout << "No model loaded. Please train or load a model first." << std::endl;
        std::cout << "Press Enter to continue...";
        std::cin.get();
        return;
    }
    
    // Display model architecture
    std::cout << "Model Architecture:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Input size: " << EMBEDDING_DIM << std::endl;
    std::cout << "Hidden layer size: " << HIDDEN_SIZE << std::endl;
    std::cout << "Output size: " << NUM_CLASSES << std::endl;
    std::cout << "Sequence length: " << MAX_SEQUENCE_LENGTH << std::endl;
    
    // Display vocabulary info
    std::cout << "\nVocabulary Information:" << std::endl;
    std::cout << "-----------------------" << std::endl;
    std::cout << "Total vocabulary size: " << nn->vocabSize << " words" << std::endl;
    std::cout << "Embedding dimension: " << EMBEDDING_DIM << std::endl;
    
    // Show some example words from vocabulary
    std::cout << "\nVocabulary examples:" << std::endl;
    int numExamples = std::min(20, nn->vocabSize);
    for (int i = 0; i < numExamples; i++) {
        std::cout << "  " << i << ": \"" << nn->indexToWord[i] << "\"" << std::endl;
    }
    if (nn->vocabSize > numExamples) {
        std::cout << "  ... and " << (nn->vocabSize - numExamples) << " more words" << std::endl;
    }
    
    std::cout << "\nPress Enter to continue...";
    std::cin.get();
}

// Main application function
int main() {
    // Print CUDA device information
    printDeviceInfo();
    
    // Create and initialize neural network
    TextNeuralNetwork nn;
    initializeTextNeuralNetwork(&nn);
    
    // Main menu loop
    bool running = true;
    while (running) {
        // Clear screen (implementation-dependent)
        #ifdef _WIN32
        system("cls");
        #else
        system("clear");
        #endif
        
        printHeader("CUDA TEXT CLASSIFICATION");
        
        // Show main menu
        std::cout << "1. Train new model" << std::endl;
        std::cout << "2. Load existing model" << std::endl;
        std::cout << "3. Analyze text" << std::endl;
        std::cout << "4. Model information" << std::endl;
        std::cout << "5. Exit" << std::endl;
        
        // Get user choice
        int choice = getValidInput<int>("\nEnter your choice (1-5): ", 1, 5);
        
        // Process user choice
        switch (choice) {
            case 1:
                trainModelOption(&nn);
                break;
            case 2:
                loadModelOption(&nn);
                break;
            case 3:
                analyzeTextOption(&nn);
                break;
            case 4:
                showModelInfoOption(&nn);
                break;
            case 5:
                running = false;
                break;
        }
    }
    
    // Free neural network resources
    freeTextNeuralNetwork(&nn);
    
    return 0;
}
