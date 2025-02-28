#ifndef TEXT_NEURAL_NETWORK_UTILS_H
#define TEXT_NEURAL_NETWORK_UTILS_H

#include "text_neural_network.h"

/**
 * Calculate the classification accuracy
 * 
 * @param predictions Model predictions [batchSize x numClasses]
 * @param labels Target labels (integer class indices) [batchSize]
 * @param numSamples Number of samples
 * @return Accuracy as a float between 0 and 1
 */
float calculateAccuracy(const float* predictions, const int* labels, int numSamples);

/**
 * Convert predicted probabilities to class labels
 * 
 * @param predictions Model predictions [batchSize x numClasses]
 * @param labels Output labels (integer class indices) [batchSize]
 * @param numSamples Number of samples
 */
void predictionsToLabels(const float* predictions, int* labels, int numSamples);

/**
 * Print model performance metrics
 * 
 * @param predictions Model predictions [batchSize x numClasses]
 * @param labels True labels (integer class indices) [batchSize]
 * @param numSamples Number of samples
 */
void printMetrics(const float* predictions, const int* labels, int numSamples);

/**
 * Print a confusion matrix for binary classification
 * 
 * @param predictions Model predictions [batchSize x numClasses]
 * @param labels True labels (integer class indices) [batchSize]
 * @param numSamples Number of samples
 */
void printConfusionMatrix(const float* predictions, const int* labels, int numSamples);

/**
 * Save trained model weights to files
 * 
 * @param nn Pointer to the neural network
 * @param prefix Prefix for output files
 */
void saveModelWeights(TextNeuralNetwork* nn, const std::string& prefix);

/**
 * Load model weights from files
 * 
 * @param nn Pointer to the neural network
 * @param prefix Prefix for input files
 */
void loadModelWeights(TextNeuralNetwork* nn, const std::string& prefix);

/**
 * Save vocabulary to a file
 * 
 * @param nn Pointer to the neural network
 * @param filename Output file name
 */
void saveVocabulary(TextNeuralNetwork* nn, const std::string& filename);

/**
 * Load vocabulary from a file
 * 
 * @param nn Pointer to the neural network
 * @param filename Input file name
 */
void loadVocabulary(TextNeuralNetwork* nn, const std::string& filename);

#endif // TEXT_NEURAL_NETWORK_UTILS_H
