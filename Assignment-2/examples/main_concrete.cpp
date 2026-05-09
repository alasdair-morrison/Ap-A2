#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <random>

#include "../include/sklearn_cpp/linear_model/utils.hpp"
#include "../include/sklearn_cpp/linear_model/LinearRegression.hpp"

using sklearn_cpp::linear_model::LinearRegression;
using sklearn_cpp::utils::StandardScaler;

// Shuffle data and labels together
void shuffleData(
    std::vector<std::vector<double>>& xData,
    std::vector<double>& yData,
    unsigned int seed = 42)
{
    if (xData.size() != yData.size()) {
        throw std::invalid_argument("xData and yData size mismatch");
    }
    
    int totalSamples = static_cast<int>(xData.size());
    std::vector<int> indices(totalSamples);
    for (int i = 0; i < totalSamples; ++i) {
        indices[i] = i;
    }
    
    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);
    
    std::vector<std::vector<double>> shuffledX(totalSamples);
    std::vector<double> shuffledY(totalSamples);
    
    for (int i = 0; i < totalSamples; ++i) {
        shuffledX[i] = xData[indices[i]];
        shuffledY[i] = yData[indices[i]];
    }
    
    xData = shuffledX;
    yData = shuffledY;
}

// Split data into training and testing sets
void trainTestSplit(
    const std::vector<std::vector<double>>& xData,
    const std::vector<double>& yData,
    std::vector<std::vector<double>>& xTrain,
    std::vector<double>& yTrain,
    std::vector<std::vector<double>>& xTest,
    std::vector<double>& yTest,
    double testSize = 0.2)
{
    if (xData.size() != yData.size()) {
        throw std::invalid_argument("xData and yData size mismatch");
    }
    
    int totalSamples = static_cast<int>(xData.size());
    int trainSize = static_cast<int>(totalSamples * (1.0 - testSize));
    
    // Split after shuffling: first 80% for training, last 20% for testing
    for (int i = 0; i < trainSize; ++i) {
        xTrain.push_back(xData[i]);
        yTrain.push_back(yData[i]);
    }
    
    for (int i = trainSize; i < totalSamples; ++i) {
        xTest.push_back(xData[i]);
        yTest.push_back(yData[i]);
    }
}

// Calculate Mean Squared Error
double calculateMSE(
    const std::vector<double>& yActual,
    const std::vector<double>& yPredicted)
{
    if (yActual.size() != yPredicted.size()) {
        throw std::invalid_argument("Size mismatch");
    }
    
    double sumSquaredErrors = 0.0;
    int numSamples = static_cast<int>(yActual.size());
    
    for (int i = 0; i < numSamples; ++i) {
        double error = yPredicted[i] - yActual[i];
        sumSquaredErrors += error * error;
    }
    
    return sumSquaredErrors / numSamples;
}

// Calculate Mean Absolute Error
double calculateMAE(
    const std::vector<double>& yActual,
    const std::vector<double>& yPredicted)
{
    if (yActual.size() != yPredicted.size()) {
        throw std::invalid_argument("Size mismatch");
    }
    
    double sumAbsoluteErrors = 0.0;
    int numSamples = static_cast<int>(yActual.size());
    
    for (int i = 0; i < numSamples; ++i) {
        double error = std::abs(yPredicted[i] - yActual[i]);
        sumAbsoluteErrors += error;
    }
    
    return sumAbsoluteErrors / numSamples;
}

int main() {
    try {
        // Load data
        std::cout << "Loading concrete.csv..." << std::endl;
        Database dataset("data/concrete.csv");
        
        auto xData = dataset.getData();
        auto yData = dataset.getLabels();
        
        int totalSamples = static_cast<int>(xData.size());
        int numFeatures = static_cast<int>(xData[0].size());
        
        std::cout << "Loaded " << totalSamples << " samples with " 
                  << numFeatures << " features" << std::endl << std::endl;
        
        // Split data
        std::cout << "Splitting data (80/20)..." << std::endl;
        std::vector<std::vector<double>> xTrain;
        std::vector<double> yTrain;
        std::vector<std::vector<double>> xTest;
        std::vector<double> yTest;
        
        trainTestSplit(xData, yData, xTrain, yTrain, xTest, yTest, 0.2);
        
        int numTrainSamples = static_cast<int>(xTrain.size());
        int numTestSamples = static_cast<int>(xTest.size());
        
        std::cout << "Train: " << numTrainSamples << " | Test: " 
                  << numTestSamples << std::endl << std::endl;
        
        // Scale features
        std::cout << "Scaling features..." << std::endl;
        StandardScaler scaler;
        auto xTrainScaled = scaler.fitTransform(xTrain);
        auto xTestScaled = scaler.transform(xTest);
        std::cout << "Done" << std::endl << std::endl;
        
        // Train model with extreme iteration count to test convergence limits
        std::cout << "Training model with 100,000 iterations..." << std::endl;
        std::cout << "(Learning rate: 0.005, extreme iteration testing)" << std::endl;
        LinearRegression model(0.005, 100000, 1e-8, true, 5000);
        model.fit(xTrainScaled, yTrain);
        
        std::cout << "Model trained" << std::endl;
        std::cout << "Bias: " << std::fixed << std::setprecision(4) 
                  << model.getBias() << std::endl << std::endl;
        
        // Predict
        std::cout << "Making predictions..." << std::endl;
        auto testPredictions = model.predict(xTestScaled);
        std::cout << "Done" << std::endl << std::endl;
        
        // Evaluate
        double mse = calculateMSE(yTest, testPredictions);
        double mae = calculateMAE(yTest, testPredictions);
        double r2 = sklearn_cpp::utils::r2Score(yTest, testPredictions);
        
        std::cout << "Results:" << std::endl;
        std::cout << "MSE: " << std::fixed << std::setprecision(4) << mse << std::endl;
        std::cout << "MAE: " << std::fixed << std::setprecision(4) << mae << std::endl;
        std::cout << "R2:  " << std::fixed << std::setprecision(4) << r2 << std::endl << std::endl;
        
        // Show some predictions
        std::cout << "Sample predictions (first 10 test samples):" << std::endl;
        std::cout << std::setw(8) << "Sample" 
                  << std::setw(12) << "Predicted" 
                  << std::setw(12) << "Actual"
                  << std::setw(10) << "Error" << std::endl;
        std::cout << std::string(42, '-') << std::endl;
        
        int numSamplesToDisplay = std::min(10, numTestSamples);
        for (int i = 0; i < numSamplesToDisplay; ++i) {
            double error = testPredictions[i] - yTest[i];
            std::cout << std::setw(8) << i + 1
                      << std::setw(12) << std::fixed << std::setprecision(2) 
                      << testPredictions[i]
                      << std::setw(12) << std::fixed << std::setprecision(2) 
                      << yTest[i]
                      << std::setw(10) << std::fixed << std::setprecision(2) 
                      << error << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
