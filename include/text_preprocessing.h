#ifndef TEXT_PREPROCESSING_H
#define TEXT_PREPROCESSING_H

#include "text_neural_network.h"
#include <string>
#include <vector>

// Function declarations for text preprocessing

/**
 * Tokenize a text string into words
 * 
 * @param text The input text to tokenize
 * @return A vector of token strings
 */
std::vector<std::string> tokenize(const std::string& text);

/**
 * Build vocabulary from a collection of texts
 * This function counts word frequencies and adds the most frequent words
 * to the vocabulary up to VOCABULARY_SIZE
 * 
 * @param nn Pointer to the neural network
 * @param texts Vector of text samples to process
 */
void buildVocabulary(TextNeuralNetwork* nn, const std::vector<std::string>& texts);

/**
 * Preprocess a batch of texts into word indices
 * Converts words to their vocabulary indices and pads sequences
 * 
 * @param nn Pointer to the neural network
 * @param texts Vector of text samples to process
 * @param indices Output array for word indices
 */
void preprocessBatch(TextNeuralNetwork* nn, const std::vector<std::string>& texts, int* indices);

/**
 * Load sentiment dataset from a file
 * File format: each line has text and label separated by tab
 * 
 * @param texts Output vector of text samples
 * @param labels Output vector of sentiment labels (0 or 1)
 * @param filename Path to the dataset file
 */
void loadSentimentDataset(std::vector<std::string>& texts, std::vector<int>& labels, const std::string& filename);

/**
 * Generate a simple sentiment dataset for testing
 * Creates positive and negative examples with predictable patterns
 * 
 * @param texts Output vector of text samples
 * @param labels Output vector of sentiment labels (0 or 1)
 * @param numSamples Number of samples to generate
 */
void generateSimpleSentimentDataset(std::vector<std::string>& texts, std::vector<int>& labels, int numSamples);

#endif // TEXT_PREPROCESSING_H
