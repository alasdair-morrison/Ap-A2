#pragma once
#include "utils.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
class LogisticRegression {
    private:
        Dataset data; // Original dataset
        Dataset testData; // Testing dataset
        Dataset trainData; // Training dataset
        std::vector<std::vector<double>> weights; // Model weights
        int epochs; // Number of training iterations
        std::vector<double> biases; // Model bias
        int numClasses; // Number of classes in the dataset
        int numFeatures; // Number of features in the dataset
        int numSamples; // Number of samples in the dataset
        std::vector<double> featureMaxes; // Normalization scaling factors
        double learningRate = 0.01; // Learning rate for gradient descent
        double convergenceThreshold = 1e-3; // Threshold for convergence

        std::vector<Dataset> splitData(double trainSize, Dataset& data, Dataset& trainData, Dataset& testData) {
            std::vector<std::vector<double>> features = data.getLabels(); // Get features from the dataset
            std::vector<double> labels = data.getData(); // Get labels from the dataset
            int totalSize = features.size(); // Total number of samples in the dataset
            int trainSizeCount = static_cast<int>(totalSize * trainSize); // Calculate the number of samples for training based on the specified train size
            
            // Shuffle data before splitting
            std::vector<int> indices(totalSize);
            std::iota(indices.begin(), indices.end(), 0);
            std::srand(42); // fixed seed for reproducibility
            for (int i = totalSize - 1; i > 0; --i) {
                int j = std::rand() % (i + 1);
                std::swap(indices[i], indices[j]);
            }

            std::vector<std::vector<double>> shuffledFeatures(totalSize);
            std::vector<double> shuffledLabels(totalSize);
            for (int i = 0; i < totalSize; ++i) {
                shuffledFeatures[i] = features[indices[i]];
                shuffledLabels[i] = labels[indices[i]];
            }

            // Normalize features internally using MaxAbsScaler
            if(totalSize > 0 && shuffledFeatures[0].size() > 0) {
                size_t num_f = shuffledFeatures[0].size();
                this->featureMaxes.assign(num_f, 0.0);
                for (int i = 0; i < totalSize; ++i) {
                    for (size_t j = 0; j < num_f; ++j) {
                        this->featureMaxes[j] = std::max(this->featureMaxes[j], std::abs(shuffledFeatures[i][j]));
                    }
                }
                for (int i = 0; i < totalSize; ++i) {
                    for (size_t j = 0; j < num_f; ++j) {
                        if (this->featureMaxes[j] > 0) {
                            shuffledFeatures[i][j] /= this->featureMaxes[j];
                        }
                    }
                }
            }

            // Split the features and labels into training and testing sets
            std::vector<std::vector<double>> trainFeatures(shuffledFeatures.begin(), shuffledFeatures.begin() + trainSizeCount);
            std::vector<double> trainLabels(shuffledLabels.begin(), shuffledLabels.begin() + trainSizeCount);
            std::vector<std::vector<double>> testFeatures(shuffledFeatures.begin() + trainSizeCount, shuffledFeatures.end());
            std::vector<double> testLabels(shuffledLabels.begin() + trainSizeCount, shuffledLabels.end());
            // Return the training and testing datasets as Dataset objects
            return {Dataset(trainFeatures, trainLabels), Dataset(testFeatures, testLabels)};
        }
        
        std::vector<double> softmax(const std::vector<double>& z) {
            // Implement the softmax function to convert logits to probabilities
            std::vector<double> probabilities(this->numClasses, 0.0);
            // Find the maximum value in z for numerical stability
            double maxZ = *std::max_element(z.begin(), z.end());
            
            double sumExp = 0.0;
            
            // Calculate exponentials and keep a running total
            for (int j = 0; j < this->numClasses; ++j) {
                // Subtract maxZ to prevent overflow
                probabilities[j] = std::exp(z[j] - maxZ); 
                sumExp += probabilities[j];
            }
            
            // Divide each probability by the sum to get final percentages
            for (int j = 0; j < this->numClasses; ++j) {
                probabilities[j] /= sumExp;
            }
            return probabilities; // Return the computed probabilities
        }
        
        std::vector<double> forwardProb(std::vector<std::vector<double>> W, std::vector<double> X, std::vector<double> b) {
            std::vector<double> z(this->numClasses, 0.0); // Initialize a vector to store the logits for each class
            for (int j = 0; j < this->numClasses; ++j) {
                double dotProduct = 0.0;
        
                // Multiply each feature by its corresponding weight for this class
                for (int i = 0; i < this->numFeatures; ++i) {
                    dotProduct += X[i] * W[i][j];
                }
        
                // Add the bias for this class
                z[j] = dotProduct + b[j];
            }
            return softmax(z); // Return the probability of the positive class
        }

        double calculateLossFast(const std::vector<double>& predictedProbs, int trueClassIndex) {
            // No loop needed! Just grab the probability of the correct class.
            return -std::log(predictedProbs[trueClassIndex] + 1e-9); 
        }

        void calculateGradients(const std::vector<double>& x, 
                        const std::vector<double>& predictedProbs, 
                        int trueClassIndex,
                        std::vector<std::vector<double>>& dW, 
                        std::vector<double>& db) {
            //Calculate the error signal (dz)
            std::vector<double> dz = predictedProbs; 
            
            // subtract 1.0 ONLY from the correct class
            dz[trueClassIndex] -= 1.0; 

            // Calculate and accumulate dW and db
            for (int j = 0; j < this->numClasses; ++j) {
                
                // The bias gradient is just the error signal
                db[j] += dz[j];
                
                // The weight gradient is the feature value multiplied by the error signal
                for (int i = 0; i < this->numFeatures; ++i) {
                    dW[i][j] += x[i] * dz[j];
                }
            }
        }

    public:
        LogisticRegression(std::string filename, int epochs, double trainSize = 0.8, int numClasses = 2) :
        data{Dataset(filename)},
        testData{Dataset({}, {})},
        trainData{Dataset({}, {})},
        epochs{epochs},
        numClasses{numClasses}
        {
            std::vector<Dataset> dataSplits = splitData(trainSize, data, trainData, testData); // Split data into training and testing sets
            this->trainData = dataSplits[0]; // Set the training dataset
            this->testData = dataSplits[1]; // Set the testing dataset
            this->numFeatures = trainData.getLabels()[0].size(); // Set the number of features based on the training data
            this->weights = std::vector<std::vector<double>>(this->numFeatures, std::vector<double>(numClasses, 0.0)); // Initialize weights to zero
            this->biases = std::vector<double>(numClasses, 0.0); // Initialize biases to zero
            this->numSamples = trainData.getLabels().size(); // Set the number of samples based on the training data
        };

        void fit() {
            // Implement the training logic for logistic regression using gradient descent
            std::cout << "Starting training for " << this->epochs << " epochs..." << std::endl; // Print a message indicating the start of training
            for (int epoch = 0; epoch < this->epochs; ++epoch) {
                double total_loss = 0.0; // Variable to accumulate total loss for the epoch
                // Loop through each training sample using Stochastic Gradient Descent targeting updates immediately
                for (int i = 0; i < this->numSamples; ++i) {
                    // Initialize gradients for weights and biases
                    std::vector<std::vector<double>> dW(this->numFeatures, std::vector<double>(this->numClasses, 0.0));
                    std::vector<double> db(this->numClasses, 0.0);
                    
                    const std::vector<double>& x = trainData.getLabels()[i]; // Get the features of the current sample by reference
                    double trueClassIndex = trainData.getData()[i]; // Get the true class label of the current sample
                    
                    // Forward pass: compute predicted probabilities
                    std::vector<double> predictedProbs = forwardProb(this->weights, x, this->biases);
                    total_loss += calculateLossFast(predictedProbs, int(trueClassIndex));
                    // Calculate gradients based on the predicted probabilities and true class label
                    calculateGradients(x, predictedProbs, int(trueClassIndex), dW, db);
                    
                    // Update weights and biases immediately after each sample (Stochastic Gradient Descent)
                    for (int j = 0; j < this->numClasses; ++j) {
                        for (int k = 0; k < this->numFeatures; ++k) {
                            this->weights[k][j] -= learningRate * dW[k][j]; 
                        }
                        this->biases[j] -= learningRate * db[j]; 
                    }
                }
            }
        };
        
        std::vector<double> predict(Dataset newTestData = Dataset({}, {})) {
            // Implement the prediction logic for logistic regression
            std::vector<double> predictions; // Vector to store predicted class labels
            if (!newTestData.getLabels().empty()) {
                // Use the provided test dataset
                // Perform predictions on the new test data
                for (auto x : newTestData.getLabels()) {
                    for (size_t j = 0; j < x.size(); ++j) {
                        if (j < this->featureMaxes.size() && this->featureMaxes[j] > 0) {
                            x[j] /= this->featureMaxes[j];
                        }
                    }
                    std::vector<double> predictedProbs = forwardProb(this->weights, x, this->biases); // Get predicted probabilities for the current sample
                    int predictedClass = std::distance(predictedProbs.begin(), std::max_element(predictedProbs.begin(), predictedProbs.end())); // Determine the class with the highest probability
                    predictions.push_back(predictedClass); // Store the predicted class label
                }
            } else {
                // Perform predictions on the existing test data
                for (const auto& x : testData.getLabels()) {
                    std::vector<double> predictedProbs = forwardProb(this->weights, x, this->biases); // Get predicted probabilities for the current sample
                    int predictedClass = std::distance(predictedProbs.begin(), std::max_element(predictedProbs.begin(), predictedProbs.end())); // Determine the class with the highest probability
                    predictions.push_back(predictedClass); // Store the predicted class label
                }

            }
            return predictions; // Return the vector of predicted class labels
        };

        Dataset getTestData() const {
            return this->testData; // Return the testing dataset
        };

};

inline double calculateAccuracy(const std::vector<double>& predictions, const std::vector<double>& trueLabels) {
    int correct = 0; // Counter for correct predictions
    int total = predictions.size(); // Total number of predictions
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == trueLabels[i]) {
            ++correct; // Increment correct counter if the prediction matches the true label
        }
    }
    return static_cast<double>(correct) / total; // Return the accuracy as a percentage
}