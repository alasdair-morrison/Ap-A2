#include "../include/sklearn_cpp/linear_model/LogisticRegression.hpp"
#include "../include/sklearn_cpp/linear_model/utils.hpp"
#include <iostream>
#include <vector>
// Compile with: g++ -O3 -I ../include main_mnist.cpp -o main_mnist && ./main_mnist

int main() {
    LogisticRegression model("data/mnist_micro.csv", 30, 0.8, 10); // Create a LogisticRegression model with the specified dataset, number of epochs, train size, and number of classes
    std::cout << "Training the model..." << std::endl; // Print a message indicating that training is starting
    model.fit(); // Train the model using the training data
    std::cout << "Making predictions..." << std::endl; // Print a message indicating that predictions are being made
    Dataset MNISTTestData("data/mnist_mini.csv"); // Load the test dataset
    std::vector<double> predictions = model.predict(MNISTTestData); // Get predictions on the test data
    Dataset testData = model.getTestData(); // Retrieve the test dataset
    std::vector<double> trueLabels = testData.getData(); // Get the true labels from the test dataset
    double accuracy = 100 * accuracy_score(MNISTTestData.getData(), predictions); // Calculate the accuracy of the model
    std::cout << "Accuracy: " << accuracy << "%" << std::endl; // Print the accuracy to the console
    return 0;
}