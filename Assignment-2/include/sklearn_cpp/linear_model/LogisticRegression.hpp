#pragma once
#include "utils.hpp"
#include <string>
#include <vector>
class LogisticRegression {
    private:
        Database data; // Original dataset
        Database testData = Database(); // Testing dataset
        Database trainData = Database(); // Training dataset
        std::vector<std::vector<double>> weights; // Model weights
        int epochs; // Number of training iterations

        void splitData(double trainSize, Database& data, Database& trainData, Database& testData) {
            std::vector<std::vector<double>> features = data.getData(); // Get features from the dataset
            std::vector<int> labels = data.getLabels(); // Get labels from the dataset
            int totalSize = features.size(); // Total number of samples in the dataset
            int trainSizeCount = static_cast<int>(totalSize * trainSize); // Calculate the number of samples for training based on the specified train size
            // Split the features and labels into training and testing sets
            std::vector<std::vector<double>> trainFeatures(features.begin(), features.begin() + trainSizeCount);
            std::vector<int> trainLabels(labels.begin(), labels.begin() + trainSizeCount);
            std::vector<std::vector<double>> testFeatures(features.begin() + trainSizeCount, features.end());
            std::vector<int> testLabels(labels.begin() + trainSizeCount, labels.end());
            // Create Database objects for training and testing datasets
            this->trainData = Database(trainFeatures, trainLabels);
            this->testData = Database(testFeatures, testLabels);
        }

    public:
        LogisticRegression(std::string filename, int epochs, double trainSize = 0.8) {
            this->data = Database(filename); // Load data from file
            splitData(trainSize, data, trainData, testData); // Split data into training and testing sets
            this->epochs = epochs; // Set the number of training iterations
        };

        void fit() {
            // Implement the training logic for logistic regression
        };
        std::vector<int> predict(const std::vector<std::vector<double>>& X) {
            // Implement the prediction logic for logistic regression
            return std::vector<int>(); // Placeholder return statement
        };
};