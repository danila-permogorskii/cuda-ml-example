#include "../include/text_preprocessing.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <random>

// Tokenize a text string into words
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string word;
    std::istringstream ss(text);
    
    // Split by whitespace
    while (ss >> word) {
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), 
                      [](unsigned char c) { return std::tolower(c); });
        
        // Remove punctuation from the beginning and end
        while (!word.empty() && std::ispunct(word.front())) {
            word.erase(0, 1);
        }
        while (!word.empty() && std::ispunct(word.back())) {
            word.pop_back();
        }
        
        // Add if not empty
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }
    
    return tokens;
}

// Build vocabulary from a collection of texts
void buildVocabulary(TextNeuralNetwork* nn, const std::vector<std::string>& texts) {
    std::cout << "Building vocabulary from " << texts.size() << " texts..." << std::endl;
    
    // Keep track of word frequency
    std::unordered_map<std::string, int> wordCount;
    
    // Count words in all texts
    for (const auto& text : texts) {
        std::vector<std::string> tokens = tokenize(text);
        for (const auto& token : tokens) {
            wordCount[token]++;
        }
    }
    
    std::cout << "Found " << wordCount.size() << " unique words" << std::endl;
    
    // Convert to vector of pairs for sorting by frequency
    std::vector<std::pair<std::string, int>> wordFreq(wordCount.begin(), wordCount.end());
    std::sort(wordFreq.begin(), wordFreq.end(), 
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Add top words to vocabulary (up to VOCABULARY_SIZE - special tokens)
    int availableSlots = VOCABULARY_SIZE - nn->vocabSize;
    int addedWords = 0;
    
    for (const auto& pair : wordFreq) {
        if (addedWords >= availableSlots) break;
        
        const std::string& word = pair.first;
        if (nn->wordToIndex.find(word) == nn->wordToIndex.end()) {
            nn->wordToIndex[word] = nn->vocabSize;
            nn->indexToWord.push_back(word);
            nn->vocabSize++;
            addedWords++;
        }
    }
    
    std::cout << "Built vocabulary with " << nn->vocabSize << " words" << std::endl;
    
    // Print a few examples of the most frequent words
    std::cout << "Most frequent words: ";
    for (int i = 0; i < std::min(10, addedWords); i++) {
        std::cout << wordFreq[i].first << " (" << wordFreq[i].second << "), ";
    }
    std::cout << "..." << std::endl;
}

// Preprocess a batch of texts into word indices
void preprocessBatch(TextNeuralNetwork* nn, const std::vector<std::string>& texts, int* indices) {
    // Process each text in the batch
    for (int i = 0; i < texts.size() && i < BATCH_SIZE; i++) {
        // Tokenize the text
        std::vector<std::string> tokens = tokenize(texts[i]);
        
        // Convert tokens to indices
        for (int j = 0; j < tokens.size() && j < MAX_SEQUENCE_LENGTH; j++) {
            const std::string& token = tokens[j];
            // Look up token in vocabulary, use UNK if not found
            int idx = nn->wordToIndex.count(token) ? 
                nn->wordToIndex[token] : nn->wordToIndex[UNK_TOKEN];
            indices[i * MAX_SEQUENCE_LENGTH + j] = idx;
        }
        
        // Pad with PAD_TOKEN
        for (int j = tokens.size(); j < MAX_SEQUENCE_LENGTH; j++) {
            indices[i * MAX_SEQUENCE_LENGTH + j] = nn->wordToIndex[PAD_TOKEN];
        }
    }
}

// Load sentiment dataset from a file
void loadSentimentDataset(std::vector<std::string>& texts, std::vector<int>& labels, const std::string& filename) {
    texts.clear();
    labels.clear();
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open dataset file: " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Find the tab separator between text and label
        size_t tabPos = line.find('\t');
        if (tabPos == std::string::npos) continue;
        
        // Extract text and label
        std::string text = line.substr(0, tabPos);
        int label = std::stoi(line.substr(tabPos + 1));
        
        texts.push_back(text);
        labels.push_back(label);
    }
    
    std::cout << "Loaded " << texts.size() << " samples from " << filename << std::endl;
}

// Generate a simple sentiment dataset for testing
void generateSimpleSentimentDataset(std::vector<std::string>& texts, std::vector<int>& labels, int numSamples) {
    texts.clear();
    labels.clear();
    
    // Positive and negative word lists
    std::vector<std::string> positiveWords = {
        "good", "great", "excellent", "wonderful", "amazing", "fantastic", 
        "terrific", "outstanding", "superb", "brilliant", "awesome", "love",
        "happy", "joy", "pleased", "delighted", "impressed", "satisfied"
    };
    
    std::vector<std::string> negativeWords = {
        "bad", "terrible", "awful", "horrible", "disappointing", "poor", 
        "mediocre", "frustrating", "sad", "angry", "annoyed", "hate",
        "disliked", "failure", "worst", "wasted", "regret", "unfortunate"
    };
    
    // Neutral filler words
    std::vector<std::string> fillerWords = {
        "the", "a", "an", "this", "that", "these", "those", "it", "they",
        "was", "is", "are", "were", "been", "being", "have", "has", "had",
        "very", "quite", "really", "so", "much", "many", "some", "any",
        "movie", "film", "show", "product", "experience", "service", "book",
        "read", "watch", "use", "buy", "recommend", "think", "feel", "believe"
    };
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> labelDist(0, 1); // 0 for negative, 1 for positive
    std::uniform_int_distribution<> wordCountDist(5, 20); // Words per review
    std::uniform_int_distribution<> sentimentWordDist(1, 5); // Sentiment words per review
    std::uniform_int_distribution<> posWordIdx(0, positiveWords.size() - 1);
    std::uniform_int_distribution<> negWordIdx(0, negativeWords.size() - 1);
    std::uniform_int_distribution<> fillerWordIdx(0, fillerWords.size() - 1);
    
    for (int i = 0; i < numSamples; i++) {
        int label = labelDist(gen);
        labels.push_back(label);
        
        // Create the sample text
        std::string text;
        int wordCount = wordCountDist(gen);
        int sentimentWords = sentimentWordDist(gen);
        
        // Add sentiment words based on the label
        for (int w = 0; w < wordCount; w++) {
            if (w > 0) text += " ";
            
            if (w < sentimentWords) {
                // Add a sentiment word
                if (label == 1) { // Positive
                    text += positiveWords[posWordIdx(gen)];
                } else { // Negative
                    text += negativeWords[negWordIdx(gen)];
                }
            } else {
                // Add a filler word
                text += fillerWords[fillerWordIdx(gen)];
            }
        }
        
        texts.push_back(text);
    }
    
    std::cout << "Generated " << texts.size() << " synthetic samples for testing" << std::endl;
    std::cout << "Examples:" << std::endl;
    for (int i = 0; i < std::min(5, (int)texts.size()); i++) {
        std::cout << "  [" << labels[i] << "] " << texts[i] << std::endl;
    }
}
