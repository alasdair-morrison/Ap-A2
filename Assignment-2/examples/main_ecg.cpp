#include "../include/sklearn_cpp/linear_model/LogisticRegression.hpp"
#include "../include/sklearn_cpp/linear_model/utils.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Starting Logistic Regression on ECG dataset..." << std::endl; // Print a message indicating the start of the process
    LogisticRegression model("data/ecg.csv", 30); // Create a LogisticRegression model with the specified dataset and number of epochs
    std::cout << "Training the model..." << std::endl; // Print a message indicating that training is starting
    model.fit(); // Train the model using the training data
    std::cout << "Making predictions..." << std::endl; // Print a message indicating that predictions are being made
    std::vector<double> predictions = model.predict(); // Get predictions on the test data
    Database testData = model.getTestData(); // Retrieve the test dataset
    std::vector<double> trueLabels = testData.getLabels(); // Get the true labels from the test dataset
    double accuracy = calculateAccuracy(predictions, trueLabels); // Calculate the accuracy of the model
    std::cout << "Accuracy: " << accuracy << std::endl; // Print the accuracy to the console
    return 0;
}